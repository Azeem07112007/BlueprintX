"""Quick smoke test for the room-polygon 3D builder."""
import sys
sys.path.insert(0, r"D:\BlueprintX")

from pipeline import BlueprintPipeline
from backend.engine.model3d.obj_exporter import ObjExporter
import os

res = BlueprintPipeline().run("sample_floorplan.png")
m   = res.building.model_3d
b   = res.building

print(f"Rooms:     {len(b.rooms)}")
print(f"Vertices:  {m.vertex_count}")
print(f"Triangles: {m.triangle_count}")
print(f"Y range:   {m.vertices[:,1].min():.2f} to {m.vertices[:,1].max():.2f} m")
print(f"X range:   {m.vertices[:,0].min():.2f} to {m.vertices[:,0].max():.2f} m")
print(f"Z range:   {m.vertices[:,2].min():.2f} to {m.vertices[:,2].max():.2f} m")

from backend.engine.model3d.glb_exporter import GlbExporter
out_glb = "building_3d.glb"
GlbExporter().export(m, out_glb)
if os.path.exists(out_glb):
    print(f"GLB size:  {os.path.getsize(out_glb)/1024:.1f} KB")

out = "building_v3_roomwalls.obj"
ObjExporter().export(m, out)
print(f"OBJ size:  {os.path.getsize(out)/1024:.1f} KB")
print("OK")
