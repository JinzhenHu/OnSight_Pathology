import os
import glob
import pathlib
import shutil
import random

# the dir that contains the jpg files and the .txt files generated from step2

labels_dir = r"C:\Users\Shadow\Desktop\TEMP_TESTING_FOLDER\out"

train_dir = r'C:\Users\Shadow\Desktop\TEMP_TESTING_FOLDER\out_train'

train_split = 0.85
##########

images = glob.glob(os.path.join(labels_dir, "*.jpg"))

random.shuffle(images)
images_train = images[:int(train_split * len(images))]
images_test = images[int(train_split * len(images)):]


labels_train_dir = os.path.join(train_dir, "labels", "train")
labels_test_dir = os.path.join(train_dir, "labels", "test")
images_train_dir = os.path.join(train_dir, "images", "train")
images_test_dir = os.path.join(train_dir, "images", "test")

pathlib.Path(labels_train_dir).mkdir(parents=True)
pathlib.Path(labels_test_dir).mkdir(parents=True)
pathlib.Path(images_train_dir).mkdir(parents=True)
pathlib.Path(images_test_dir).mkdir(parents=True)

for fp in images_train:
    fp_name = os.path.basename(fp).split('.')[0]

    shutil.copy(fp, images_train_dir)
    shutil.copy(os.path.join(labels_dir, fp_name + ".txt"), labels_train_dir)

for fp in images_test:
    fp_name = os.path.basename(fp).split('.')[0]

    shutil.copy(fp, images_test_dir)
    shutil.copy(os.path.join(labels_dir, fp_name + ".txt"), labels_test_dir)

