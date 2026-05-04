import sys
import os
import itertools
import time
import yaml
import cv2
import numpy as np
import torch
import mss
from PIL import Image, ImageDraw
from ultralytics.utils.plotting import Annotator
from mouse_click import get_screen_region
from Model import MyMitosisDetection
from utlis.detection_helper import create_anchors, process_output, rescale_boxes
from utlis.nms_WSI import nms, nms_patch
import torchvision.transforms as T
#import timm
# Set up device and model configuration
device = torch.device("cuda")
print(device)

model_path = "./retinanet/bestmodel.pth"
config_path = "./retinanet/file/config.yaml"

mapping = {0:"15x",
           1:"20x",
           2:"40x"}
Transform =T.Compose([
    T.ToTensor(),
    T.CenterCrop(224), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Detection parameters
thred = 0.1
Cl = 0.1
ZOOM_FACTOR = 1

detector = MyMitosisDetection(model_path, config, Cl, thred)
vit_magnification_detector  = torch.load(r'D:\JHU\Phedias\HAVOC\CNN\weights\best_3class.pth',weights_only= False)
model = detector.load_model()


def Mitosis_detection(region):
    """
    Captures a screen region, processes image tiles for mitosis detection,
    annotates the detections, and displays the result in a window.
    """
    # with mss.mss() as sct, torch.no_grad():
    cv2.namedWindow("Mitosis Detection", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Confidence", "Mitosis Detection", 20, 100, lambda x: None)
    with mss.mss() as sct, torch.no_grad():
        while True:
            screenshot = sct.grab(region)
            frame_orig = np.array(screenshot, dtype=np.uint8)
            frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR)

            #Magnification Detector
            # tile_downsample = Transform(frame).to(device).unsqueeze(0)
            # output = vit_magnification_detector(tile_downsample)
            # probabilities = torch.softmax(output,dim=1).cpu().numpy()[0]     
            # label =  np.argmax(probabilities)

            # Define tile parameters
            tile_size = 512
            stride = 384  # 25% overlap
            all_detections = []
            all_magnification =[]
            start_time = time.time()

            h, w = frame.shape[:2]
            num_x = (w - tile_size) // stride + 1
            num_y = (h - tile_size) // stride + 1

            # Iterate over the tiles in the frame
            for i in range(num_x):
                for j in range(num_y):
                    x_start = i * stride
                    y_start = j * stride
                    tile = frame[y_start:y_start + tile_size, x_start:x_start + tile_size, :]

                    box = detector.process_image(tile)
                    if box.size == 0:
                        continue
                    # tile_downsample = Transform(frame).to(device).unsqueeze(0)
                    # output = vit_magnification_detector(tile_downsample)
                    # probabilities = torch.softmax(output,dim=1).cpu().numpy()[0] 
                    # all_magnification.append(probabilities)    


                    # Adjust box coordinates to the global frame
                    box_coords = np.atleast_2d(box)
                    global_boxes = box_coords.copy()
                    global_boxes[:, 0] += x_start
                    global_boxes[:, 1] += y_start
                    global_boxes[:, 2] += x_start
                    global_boxes[:, 3] += y_start
                    all_detections.append(global_boxes.tolist())

            # Apply Non-Maximum Suppression on all detections
            prediction = nms(list(itertools.chain(*all_detections)))
            # final_prob = np.mean(all_magnification,axis=0)
            # label =  np.argmax(final_prob)
            # Set up the annotator for drawing boxes
            annotator = Annotator(
                frame,
                line_width=2,
                font_size=14,
                font="Arial_Bold.ttf",
                pil=False,
                example="abc",
            )
            conf_thresh = cv2.getTrackbarPos("Confidence", "Mitosis Detection") / 100

            # Annotate detections that meet the confidence threshold
            for row in prediction:
                if row[5] >= conf_thresh:
                    # Choose color based on confidence
                    if row[5] > 0.7:
                        base_color = (0, 255, 0)
                    elif row[5] > 0.4:
                        base_color = (255, 0, 0)
                    else:
                        base_color = (0, 0, 255)
                    label_text = f'score {row[5]*100:.1f} %'
                    annotator.box_label(
                        row[:4],
                        label=label_text,
                        color=base_color,
                        txt_color=(255, 255, 255),
                        rotated=False,
                    )
                    # Draw a confidence bar
                    bar_height = 4
                    bar_width = int(row[2] - row[0])
                    cv2.rectangle(
                        frame,
                        (int(row[0]), int(row[3])),
                        (int(row[0] + bar_width * row[5]), int(row[3]) + bar_height),
                        base_color,
                        -1
                    )

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Calculate text position
            # text = f"Magnification: {mapping[label]}"
            # (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # # Set position with right margin (10px from right edge)
            # x_pos = frame_width - text_width - 10
            # y_pos = 30  # 30px from top

            # # Draw text with background and foreground
            # cv2.putText(frame, text, (x_pos, y_pos),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Black background
            # cv2.putText(frame, text, (x_pos, y_pos),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text
            
            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            annotated_result = annotator.result()
            annotated_frame = annotated_result.astype(np.uint8)

            print(f"Inference time: {(time.time() - start_time)*1000:.3f} ms")
            cv2.imshow("image", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    region = get_screen_region()
    Mitosis_detection(region)
#    # image_path = input("Please enter the path to the image you want to analyze: ")
#     image_path = "./retinanet/mitosis_image.png"
#     #if os.path.exists(image_path):
#     screenshot = cv2.imread(image_path)
#     Mitosis_detection(screenshot)
#     # else:
#     #     print("Error: The file path does not exist.")