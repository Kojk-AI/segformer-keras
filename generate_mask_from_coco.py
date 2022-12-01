from pycocotools.coco import COCO
import numpy as np
import cv2
import os


def generate_masks(mask_dir, annFile):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    coco=COCO(annFile)

    imgids = coco.getImgIds()
    for imgid in imgids:
        anns = coco.getAnnIds(imgIds=imgid)
        first = True
        for id in anns:
            for seg in coco.loadAnns(id):
                if first == True:
                    mask = coco.annToMask(seg) * seg['category_id']
                    first = False
                else:
                    mask = np.maximum(coco.annToMask(seg) * seg['category_id'], mask)
        fname = str(imgid).zfill(12) + ".png"
        cv2.imwrite(os.path.join(mask_dir, fname),mask)

def run():
    mask_dir = "train\\mask"
    annFile = "coco-2017\\raw\\instances_train2017.json"

    generate_masks(mask_dir, annFile)

    mask_dir = "validation\\mask"
    annFile = "coco-2017\\raw\\instances_val2017.json"

    generate_masks(mask_dir, annFile)

if __name__ == "__main__":
   run()
