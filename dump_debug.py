import cv2
from pipeline import BlueprintPipeline

p = BlueprintPipeline()
res = p.run('sample_floorplan.png')

for k, img in res.debug_images.items():
    if img is not None:
        cv2.imwrite(f"debug_{k}.png", img)
print("Saved debug images.")
