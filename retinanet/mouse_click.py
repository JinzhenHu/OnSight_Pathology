import mss
import numpy as np
import cv2
from pynput import mouse
import torch
from ultralytics import YOLO
import ultralytics
from mss import mss

# Example usage

def get_screen_region():
    print("Please drag the mouse to select an area.")
    print("Press 'ESC' to cancel the selection.")

    region = {}
    selecting = {"start": None, "end": None}

    def on_click(x, y, button, pressed):
        if pressed:  # Mouse pressed event
            if selecting["start"] is None:
                selecting["start"] = (x, y)
                print(f"Start position: {selecting['start']}")
        else:  # Mouse released event
            if selecting["start"] is not None and selecting["end"] is None:
                selecting["end"] = (x, y)
                print(f"End position: {selecting['end']}")
                return False  # Stop listener after selection

    # Start listening for mouse events
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    # Calculate the region
    if selecting["start"] and selecting["end"]:
        start_x, start_y = selecting["start"]
        # end_x, end_y = selecting["end"]
        # NOTE: user just clicks top left of capture area. we will be using a 1024 window !!!!!!!!!!
        #end_x, end_y = start_x + 2048, start_y + 1024 #896
        end_x, end_y = start_x + 512 , start_y + 512 
        region = {
            "left": min(start_x, end_x),
            "top": min(start_y, end_y),
            "width": abs(end_x - start_x),
            "height": abs(end_y - start_y),
        }
        return region
    else:
        print("Selection cancelled.")
        return None
# def get_screen_region():
#     with mss() as sct:
#         screen_width = sct.monitors[1]["width"]  # Get primary monitor width
#         screen_height = sct.monitors[1]["height"]  # Get primary monitor height

#     # Start position: 1/4 from the top-left corner
#     start_x = screen_width // 4  
#     start_y = screen_height // 4 

#     # Define the fixed region size
#     region_width = 896
#     region_height = 896

#     # Define the region
#     region = {
#         "left": start_x,
#         "top": start_y,
#         "width": region_width,
#         "height": region_height,
#     }

#     #print(f"Automatically selected region: {region}")
#     return region