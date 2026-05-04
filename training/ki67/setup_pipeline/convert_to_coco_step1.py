import glob
import json
import pathlib

import openslide
import cv2
import numpy as np
from shapely.geometry import Polygon
import os

def convert(geojson_path,wsi_path,output_image_path,output_json_path):
    with open(geojson_path) as f:
        geojson_data = json.load(f)

    slide = openslide.OpenSlide(wsi_path)

    # === STEP 2: Find the rectangular crop area ===
    # Assuming it's the first annotation in GeoJSON
    region_feature = next(f for f in geojson_data["features"] if f["properties"].get("type") == "annotation")
    region_coords = region_feature["geometry"]["coordinates"][0]

    x_coords, y_coords = zip(*region_coords)
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    crop_width = x_max - x_min
    crop_height = y_max - y_min

    # === STEP 3: Read cropped region from WSI and save ===
    region = slide.read_region((x_min, y_min), 0, (crop_width, crop_height)).convert("RGB")
    region_cv = np.array(region)[..., ::-1]  # Convert to BGR for OpenCV
    cv2.imwrite(output_image_path, region_cv)

    # === STEP 4: Prepare COCO output ===
    coco = {
        "images": [{
            "id": 1,
            "file_name": output_image_path,
            "width": crop_width,
            "height": crop_height
        }],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Positive"},
            {"id": 2, "name": "Negative"}
        ]
    }

    class_to_id = {
        "Positive": 1,
        "Negative": 2
    }

    annotation_id = 1

    for feature in geojson_data["features"]:
        if feature["properties"].get("type") != "cell":
            continue

        coords = feature["geometry"]["coordinates"][0]
        polygon = Polygon(coords)

        # Only keep cells that lie inside selected region
        if not polygon.within(Polygon(region_coords)):
            continue

        # Shift polygon relative to crop origin
        shifted_coords = [[x - x_min, y - y_min] for x, y in coords]
        shifted_flat = [coord for point in shifted_coords for coord in point]
        shifted_poly = Polygon(shifted_coords)
        x, y, max_x, max_y = shifted_poly.bounds
        width = max_x - x
        height = max_y - y
        area = shifted_poly.area

        class_name = feature["properties"]["class"]
        category_id = class_to_id.get(class_name)
        if category_id is None:
            continue

        coco["annotations"].append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": category_id,
            "segmentation": [shifted_flat],  # this is key!
            "bbox": [x, y, width, height],
            "area": area,
            "iscrowd": 0
        })

        annotation_id += 1

    # Save output
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Cropped image saved to: {output_image_path}")
    print(f"COCO annotations saved to: {output_json_path}")

def visualizer(image_path, coco_json_path, output_path_obb=None, output_path_seg=None):

    # Load COCO annotations
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Map category_id to name for labeling
    category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Choose colors: BGR
    colors = {
        "Positive": (0, 0, 255),  # Red
        "Negative": (255, 0, 0)  # Blue
    }

    if output_path_obb is not None:
        # Load image
        image = cv2.imread(image_path)

        # Draw bounding boxes
        for ann in coco["annotations"]:
            x, y, w, h = map(int, ann["bbox"])
            category_id = ann["category_id"]
            label = category_map[category_id]
            color = colors.get(label, (255, 255, 255))

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            # cv2.putText(image, label[0], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save or display the result
        cv2.imwrite(output_path_obb, image)
        print(f"Annotated preview saved to: {output_path_obb}")

    # === Draw each segmentation polygon ===

    if output_path_seg is not None:
        # Load image
        image = cv2.imread(image_path)

        for ann in coco["annotations"]:
            category_id = ann["category_id"]
            label = category_map[category_id]
            color = colors.get(label, (255, 255, 255))

            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=1)
                # cv2.putText(image, label[0], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(output_path_seg, image)
        print(f"Segmentation mask preview saved to: {output_path_seg}")


if __name__ == "__main__":

    '''
    root_dir should point to a folder containing folders, where each one has a svs slide and geojson annotations.
    the first part of the if-statement will show how the annotations and coco file looks like; a sanity check.
    the second part is basically the same except everything gets saved to a single folder without any of the visualizations.
    '''

    if 1:
        root_dir = r'C:\Users\Shadow\Desktop\TEMP_TESTING_FOLDER'
        folders = next(os.walk(root_dir))[1]

        for folder in folders:
            if 'out' in folder or '_' in folder[0]: continue
            curr_folder = os.path.join(root_dir, folder)

            curr_out_folder = os.path.join(curr_folder, 'out')
            pathlib.Path(curr_out_folder).mkdir(parents=True, exist_ok=True)

            svs_path = glob.glob(os.path.join(curr_folder, "*.svs"))
            assert len(svs_path) == 1
            svs_path = svs_path[0]

            geojsons = glob.glob(os.path.join(curr_folder, "*.geojson"))
            for geojson_path in geojsons:

                convert(
                    geojson_path,
                    svs_path,
                    os.path.join(curr_out_folder, os.path.basename(geojson_path) + '.jpg'),
                    os.path.join(curr_out_folder, os.path.basename(geojson_path) + '.json')
                )

                visualizer(
                    os.path.join(curr_out_folder, os.path.basename(geojson_path) + '.jpg'),
                    os.path.join(curr_out_folder, os.path.basename(geojson_path) + '.json'),
                    None,#os.path.join(curr_out_folder, os.path.basename(geojson_path) + '_obb_annotated.jpg'),
                    os.path.join(curr_out_folder, os.path.basename(geojson_path) + '_seg_annotated.jpg')
                )

    else:
        # NOTE: THIS IS AFTER I RUN THE ABOVE AND MAKE SURE EVERYTHING LOOKS GOOD. THE BELOW WILL PREPARE IN THE FORMAT THAT ILL
        # FEED INTO THE YOLO CONVERSION SCRIPT
        root_dir = r'C:\Users\Shadow\Desktop\TEMP_TESTING_FOLDER'

        folders = next(os.walk(root_dir))[1]

        out_folder = os.path.join(root_dir, 'out')
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

        _i = 0
        for folder in folders:
            if 'out' in folder or '_' in folder[0]: continue

            curr_folder = os.path.join(root_dir, folder)

            svs_path = glob.glob(os.path.join(curr_folder, "*.svs"))
            assert len(svs_path) == 1
            svs_path = svs_path[0]

            geojsons = glob.glob(os.path.join(curr_folder, "*.geojson"))
            for geojson_path in geojsons:

                convert(
                    geojson_path,
                    svs_path,
                    os.path.join(out_folder, str(_i) + '.jpg'),
                    os.path.join(out_folder, str(_i) + '.json')
                )

                _i += 1
