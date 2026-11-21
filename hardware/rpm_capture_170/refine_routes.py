import copy
import json
import random
from pathlib import Path

import layout_work as m

root = Path(__file__).resolve().parent

p = json.loads((root / "layout.json").read_text())
template = copy.deepcopy(p["wires"])
best = None
allends = {m.point(w[k]) for w in template for k in ["a", "b"]}
for seed in range(200):
    rng = random.Random(seed)
    order = list(range(1, len(template)))
    rng.shuffle(order)
    new = copy.deepcopy(template)
    grid = set()
    for x, y in new[0]["path"]:
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            grid.add((round(x * 2) + dx, round(y * 2) + dy))
    for idx in order:
        w = new[idx]
        blocked = m.BASE | (allends - {m.point(w["a"]), m.point(w["b"])})
        # Only W1 crosses the IC body.
        r = m.route([w["a"]], [w["b"]], blocked, grid)
        if r is None:
            break
        _, _, path, cost = r
        w["path"] = [[x / 2, y / 2] for x, y in path]
        w["cost"] = cost
        for x, y in path:
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                grid.add((x + dx, y + dy))
    else:
        sets = [set(map(tuple, w["path"])) for w in new]
        hits = sum(bool(a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])
        score = (hits, sum(w["cost"] for w in new))
        if best is None or score < best[0]:
            best = (score, seed, new)
            print("best", seed, score, flush=True)
assert best
p["wires"] = best[2]
p["refinement_seed"] = best[1]
p["routing_refinements"] = 200
(root / "layout.json").write_text(json.dumps(p, indent=2))
print("DONE", best[0])
