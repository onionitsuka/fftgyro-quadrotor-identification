import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_native_gazebo_model import adapt_model, inertia_matrix, parallel_axis
from run_native_gazebo_sitl import initial_pose_enu


def test_spawn_heading_matches_offboard_north_target():
    pose = list(map(float,initial_pose_enu().split(",")))
    assert pose[:5] == [0]*5
    assert np.pi/2-pose[5] == pytest.approx(0,abs=1e-14)
    assert float(initial_pose_enu(np.pi/2).split(",")[-1]) == pytest.approx(0)


def fixture():
    model = ET.Element("model", name="x500_base")
    for name, mass, pose in [("base_link",2.,None)] + [(f"rotor_{j}",.016,"0.174 0.174 .06 0 0 0") for j in range(4)]:
        link = ET.SubElement(model,"link",name=name)
        if pose:
            ET.SubElement(link,"pose").text = pose
        inertial = ET.SubElement(link,"inertial")
        ET.SubElement(inertial,"mass").text = str(mass)
        inertia = ET.SubElement(inertial,"inertia")
        for key in ("ixx","iyy","izz"):
            ET.SubElement(inertia,key).text = ".00002" if pose else ".03"
        ET.SubElement(link,"sensor",name="imu_sensor",type="imu")
    motors = ET.fromstring('<model><plugin name="motor"><motorConstant>0.0000085</motorConstant><momentConstant>.016</momentConstant><timeConstantUp>.0125</timeConstantUp></plugin></model>')
    settings = dict(mass_kg=1.594, inertia_kg_m2=[.0169,.0201,.02616], arm_xy_m=.1768,kf=1e-5,kq=2e-7)
    return model,motors,settings


def test_composite_mass_inertia_and_native_sensor_preservation():
    base,motors,settings = fixture()
    model, manifest = adapt_model(base,motors,settings)
    total, moment, inertia = 0.,np.zeros(3),np.zeros((3,3))
    for link in model.findall("link"):
        mass = float(link.findtext("inertial/mass"))
        pose = np.array(list(map(float,link.findtext("pose","0 0 0 0 0 0").split())))[:3]
        cg = np.array(list(map(float,link.findtext("inertial/pose","0 0 0 0 0 0").split())))[:3]
        total += mass
        moment += mass*(pose+cg)
        inertia += inertia_matrix(link)+parallel_axis(mass,pose+cg)
    np.testing.assert_allclose(total,settings["mass_kg"])
    np.testing.assert_allclose(moment,0,atol=1e-14)
    np.testing.assert_allclose(inertia,np.diag(settings["inertia_kg_m2"]),atol=1e-14)
    assert len(model.findall("link/sensor")) == 5
    assert model.findtext("plugin/timeConstantUp") == ".0125"
    assert float(model.findtext("plugin/momentConstant")) == settings["kq"]/settings["kf"]
    assert base.findtext("link/inertial/mass") == "2.0"  # Original untouched.


def test_rejects_impossible_inertia():
    base,motors,settings = fixture()
    settings["inertia_kg_m2"] = [.00001]*3
    with pytest.raises(ValueError,match="not physically valid"):
        adapt_model(base,motors,settings)
