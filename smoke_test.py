"""
smoke_test.py — Blueprint-X Redesign (Milestone 2)
Runs the full headless detection + export pipeline and prints a summary.
"""
import json
from pipeline import BlueprintPipeline
from backend.engine.export.exporter import Exporter

print("=" * 50)
print("  Blueprint-X  |  Pipeline Smoke Test (M2 Redesign)")
print("=" * 50)

p = BlueprintPipeline()
print(f"[1] Running multi-layer pipeline...")
res = p.run("sample_floorplan.png")

print(f"[2] Pipeline complete.")
b = res.building
walls_count = len(b.walls)
rooms_count = len(b.rooms)
doors_count = len(b.doors)

print(f"    Walls detected  : {walls_count} total")
print(f"    Rooms found     : {rooms_count}")
print(f"    Doors           : {doors_count}")

exp = Exporter()
out_path = "test_export_m2.json"
exp.export(b, out_path)
print(f"[3] JSON exported   : {out_path}")

# Validate JSON round-trip
data = json.loads(open(out_path, encoding="utf-8").read())
assert isinstance(data["walls"], list),     "walls must be a list"
assert isinstance(data["rooms"], list),     "rooms must be a list"
assert "topology" in data,                  "topology must be exported"

if len(data["walls"]) > 0:
    first_wall = data["walls"][0]
    print(f"\n    Sample wall     : start={first_wall['start']}  end={first_wall['end']}"
          f"  type={first_wall.get('wall_type')}")

print()
print("All assertions passed.")
print("=" * 50)
