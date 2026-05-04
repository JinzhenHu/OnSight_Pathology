# #!/usr/bin/env python
# # coding: utf-8

# get_ipython().system('pip install ultralytics')

# get_ipython().system('nvidia-smi')

# import zipfile
# with zipfile.ZipFile('./out_train.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')

# get_ipython().system('pwd')


def train():
    from ultralytics import YOLO
    import ultralytics.models.yolo.classify.train as build


    # model = YOLO("yolo11l-seg.pt")
    # results = model.train(data=yaml_file, epochs=100, imgsz=128, plots=True, device=[0, 1],
    #                       # turning off some augmentation not applicable to us
    #                       hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0, scale=0, mosaic=0, erasing=0, auto_augment=None,
    #                       # PROJECT NAME (save folder in curr directory)
    #                       project='mib'
    #                       )

    # model = YOLO("yolo12x.pt")
    model = YOLO("yolo11x-seg.pt")
    results = model.train(data=yaml_file, epochs=500, imgsz=1024, plots=True, device=[0], batch=1,
                          # turning off some augmentation not applicable to us
                          hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0, scale=0, mosaic=0, erasing=0, auto_augment=None,
                          # PROJECT NAME (save folder in curr directory)
                          # project='mib_det_yolo12x_1024_qupath'
                          project='mib_seg_yolo11x_1024_qupath'
                          )


if __name__ == '__main__':
    # yolo training folder
    yaml_file = '/workspace/out_train/train.yaml'

    train()
