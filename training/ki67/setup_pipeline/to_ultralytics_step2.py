from ultralytics.data.converter import convert_coco
import json, glob, os, pathlib

# STEP 2


out_dir = r'C:\Users\Shadow\Desktop\TEMP_TESTING_FOLDER\out'

'''
THE JSON FOR THE AUGMENTED IMAGES STILL HAS THE OLD FILE_PATH IN THE FILE. SO THIS WILL TAKE CARE OF THAT
'''
if 1:
    def adjust_json_files(folder):
        for fp in glob.glob(os.path.join(folder, "*.json")):
            with open(fp) as f:
                ann = json.load(f)
                ann['images'][0]['file_name'] = os.path.join(folder, pathlib.Path(fp).stem + '.jpg')

                # '''
                # FOR GENERAL NUCLEI DETECTOR. WE WANT JUST 1 CLASS
                # '''
                # for _ann in ann['annotations']:
                #     _ann['category_id'] = 1
                #
                # ann['categories'] = [{'id': 1, 'name': 'Nuclei'}]

            with open(fp, "w") as f:
                json.dump(ann, f, indent=2)

    adjust_json_files(out_dir)

if 1:
    convert_coco(
        labels_dir=out_dir,
        # NOTE: looks like some bug...no idea. it's supposed to save the converted coco annotations into save_dir but
        # it saves the txt files to the folder above (labels_dir). anyways, these txt files are the ultralytics compatible annotations
        save_dir=out_dir + '2',
        use_segments=True, use_keypoints=False, cls91to80=False)

