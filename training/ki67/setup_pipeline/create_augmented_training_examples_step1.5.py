import json
import shutil

import cv2
import numpy as np
from shapely.geometry import shape
import os, glob, pathlib
import random

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils





def simulate_faint_stain(image, gamma=1.2, desaturation=0.4, blur_sigma=1.2):
    """
    Simulates faint/understained nuclei appearance
    - gamma > 1.0 brightens image
    - desaturation reduces stain chroma (but not fully gray)
    - blur softens edges
    """
    # Step 1: Gamma correction (brighten)
    image_float = image.astype(np.float32) / 255.0
    gamma_corrected = np.power(image_float, 1.0 / gamma)

    # Step 2: Desaturate (but keep hint of blue)
    hsv = cv2.cvtColor((gamma_corrected * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= desaturation  # Reduce saturation (keep H)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    desaturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 3: Blur to reduce edge clarity
    faint_image = cv2.GaussianBlur(desaturated, (0, 0), sigmaX=blur_sigma)

    return faint_image

def randomize_negative_nuclei_hue(image, neg_mask, hue_range=(90, 160), mix_ratio=0.3):
    """
    Randomly shifts hue of negative nuclei to simulate stain variation (blue/purple/cyan).
    - hue_range: tuple of (min_hue, max_hue) in OpenCV HSV space [0–179]
    - mix_ratio: how much to blend (0=no change, 1=full hue overwrite)
    """
    target_hue = random.randint(*hue_range)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Blend toward target hue only inside mask
    h_new = h.copy()
    h_new[neg_mask == 255] = (
            (1 - mix_ratio) * h[neg_mask == 255] + mix_ratio * target_hue
    )

    hsv_shifted = cv2.merge([h_new, s, v]).astype(np.uint8)
    result = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

    return result

def blur_negative_nuclei_only(image, neg_mask, sigma_range=(0.6, 2.0)):
    """
    Applies random Gaussian blur to negative nuclei only.
    - sigma_range: tuple of (min_sigma, max_sigma)
    """
    sigma = np.random.uniform(*sigma_range)

    # Apply Gaussian blur to whole image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)

    # Blend blurred pixels only where negative mask is 255
    output = image.copy()
    for c in range(3):  # BGR channels
        output[:, :, c] = np.where(
            neg_mask == 255,
            blurred[:, :, c],
            image[:, :, c]
        )

    return output

def augment(image, neg_mask):
    image = simulate_faint_stain(
        image,
        gamma=np.random.uniform(1.5, 2.5),
        desaturation=np.random.uniform(0.9, 1.0),
        blur_sigma=np.random.uniform(0.5, 2.0))
    image = randomize_negative_nuclei_hue(image, neg_mask, mix_ratio=1)
    image = blur_negative_nuclei_only(image, neg_mask, sigma_range=(0.5,2.0))
    return image

def get_negative_mask(coco, image_id, negative_category_id, image_shape):
    ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[negative_category_id])
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for ann in anns:
        rle = maskUtils.frPyObjects(ann['segmentation'], image_shape[0], image_shape[1])
        m = maskUtils.decode(rle)
        m = np.any(m, axis=2) if m.ndim == 3 else m
        mask[m > 0] = 255
    return mask




if __name__ == '__main__':
    '''
    after running step1.py, you'll have tile patches + coco annotations.
    create a separate folder and put them in here (the ones you want to create some augmentations for).
    after you run this, you'll get augmented tile patches + coco annotations (unchanged).
    for the ones you want to use for training, you can copy-paste these back into the "out" folder from step1
    '''

    tile_dir = r'C:\Users\Shadow\Desktop\TEMP_TESTING_FOLDER\aug_test'


    out_dir = os.path.join(tile_dir, 'augmented')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    _ii = 0
    for fp in glob.glob(os.path.join(tile_dir, '*.jpg')):

        coco_ann_fp = os.path.join(tile_dir, str(pathlib.Path(os.path.basename(fp)).with_suffix('.json')))

        coco = COCO(coco_ann_fp)


        # 1 coco file per image. so there is only image_id=1
        neg_mask = get_negative_mask(coco, image_id=1, negative_category_id=2, image_shape=(1024,1024))

        image = cv2.imread(fp)

        for _i in range(2):
            augmented = augment(image, neg_mask)


            cv2.imwrite(os.path.join(out_dir, 'aug_' + str(_ii) + '.jpg'), augmented)
            shutil.copy(coco_ann_fp, os.path.join(out_dir, 'aug_' + str(_ii) + '.json'))

            if 0:
                from convert_to_coco_step1 import visualizer

                visualizer(
                    os.path.join(out_dir, 'aug_' + str(_ii) + '.jpg'),
                    coco_ann_fp,
                    None,
                    os.path.join(out_dir, 'aug_' + str(_ii) + '_seg.jpg'),
                )

            _ii += 1





