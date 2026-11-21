"""Generate a private X500 variant from installed PX4 SDFs, without editing them.

Keep native sensors, collisions, rotor joints and motor dynamics. Match total
mass and the initial composite inertia (not merely base_link inertia).
Generated third-party SDFs belong outside Git. Rotor inertia changes slightly
with joint angle because native rotor links are not axisymmetric.
"""
from __future__ import annotations
import copy
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def parallel_axis(mass: float, position: np.ndarray) -> np.ndarray:
    return mass * (position @ position * np.eye(3) - np.outer(position, position))


def inertia_matrix(link: ET.Element) -> np.ndarray:
    node = link.find("inertial/inertia")
    get = lambda key: float(node.findtext(key, "0"))
    return np.array([[get("ixx"),get("ixy"),get("ixz")],
                     [get("ixy"),get("iyy"),get("iyz")],
                     [get("ixz"),get("iyz"),get("izz")]])


def adapt_model(base: ET.Element, motor_model: ET.Element, settings: dict) -> tuple[ET.Element, dict]:
    model = copy.deepcopy(base)
    model.set("name", "fftgyro_x500")
    base_link = model.find("link[@name='base_link']")
    a = settings["arm_xy_m"]
    # SDF FLU coordinates; same numbering as the common FRD plant.
    xy = [(a,-a), (-a,a), (a,a), (-a,-a)]
    rotors = [model.find(f"link[@name='rotor_{j}']") for j in range(4)]
    rotor_sum = np.zeros((3,3))
    first_moment = np.zeros(3)
    rotor_mass = 0.
    for rotor, (x,y) in zip(rotors, xy, strict=True):
        pose = np.array([float(s) for s in rotor.findtext("pose").split()])
        if np.any(pose[3:] != 0) or rotor.find("inertial/pose") is not None:
            raise ValueError("Rotated/offset rotor inertials need explicit transforms")
        pose[:2] = [x,y]
        rotor.find("pose").text = " ".join(f"{v:.16g}" for v in pose)
        mass = float(rotor.findtext("inertial/mass"))
        rotor_mass += mass
        first_moment += mass * pose[:3]
        rotor_sum += inertia_matrix(rotor) + parallel_axis(mass, pose[:3])
    base_mass = settings["mass_kg"] - rotor_mass
    if base_mass <= 0:
        raise ValueError("Total mass must exceed native rotor masses")
    # Choose total COM at base_link origin; do not transplant the Leptrino-frame
    # measured CG coordinates into an unrelated Gazebo base_link frame.
    base_cg = -first_moment / base_mass
    target = np.diag(settings["inertia_kg_m2"])
    base_inertia = target - rotor_sum - parallel_axis(base_mass, base_cg)
    eigenvalues = np.linalg.eigvalsh(base_inertia)
    if eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum():
        raise ValueError("Required base inertia is not physically valid")
    base_link.find("inertial/mass").text = f"{base_mass:.16g}"
    inertial_pose = base_link.find("inertial/pose")
    if inertial_pose is None:
        inertial_pose = ET.SubElement(base_link.find("inertial"), "pose")
    inertial_pose.text = " ".join(f"{v:.16g}" for v in [*base_cg,0,0,0])
    for key, i, j in [("ixx",0,0),("iyy",1,1),("izz",2,2),("ixy",0,1),("ixz",0,2),("iyz",1,2)]:
        entry = base_link.find(f"inertial/inertia/{key}")
        if entry is None:
            entry = ET.SubElement(base_link.find("inertial/inertia"), key)
        entry.text = f"{base_inertia[i,j]:.16g}"
    for plugin in motor_model.findall("plugin"):
        plugin = copy.deepcopy(plugin)
        if plugin.find("motorConstant") is not None:
            plugin.find("motorConstant").text = str(settings["kf"])
            plugin.find("momentConstant").text = str(settings["kq"] / settings["kf"])
        model.append(plugin)
    return model, dict(total_mass_kg=settings["mass_kg"], base_mass_kg=base_mass,
        base_cg_flu_m=base_cg.tolist(), composite_initial_inertia_kg_m2=target.tolist(),
        base_inertia_kg_m2=base_inertia.tolist(), rotor_total_mass_kg=rotor_mass,
        preserved="Native sensors, joints, rotor inertials, collision/visual geometry, rotor drag, rolling moment, motor time constants 12.5/25 ms, max command 1000 rad/s, slowdown 10",
        changed="Total mass/initial inertia, horizontal rotor arms, KF/KQ, total COM at base origin",
        limitations=["Native motor input map differs from measured per-motor maps and 50-ms delay",
            "Initial aggregate inertia matched; native anisotropic rotor inertia rotates with joints",
            "Native rotor height/geometry retained; CG in Leptrino coordinates not transferred",
            "Stock visual/collision mesh is not a verified 1045 propeller geometry",
            "Native joint slowdown affects angular momentum; not calibrated rotor inertia"])


def build(px4_root: Path, artifact: dict, destination: Path) -> dict:
    sources = px4_root / "Tools/simulation/gz/models"
    paths = [sources / name / "model.sdf" for name in ("x500_base", "x500")]
    model, manifest = adapt_model(*(ET.parse(path).getroot().find("model") for path in paths), artifact["settings"])
    folder = destination / "fftgyro_x500"
    folder.mkdir(parents=True, exist_ok=False)
    root = ET.Element("sdf", version="1.9")
    root.append(model)
    ET.indent(root)
    ET.ElementTree(root).write(folder / "model.sdf", encoding="utf-8", xml_declaration=True)
    manifest["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest["generated_sdf_sha256"] = hashlib.sha256((folder / "model.sdf").read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
