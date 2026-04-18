import collections
import pickle
import numpy as np
from fastai.vision.learner import create_body
from fastai.vision import models
from object_detection_fastai.models.RetinaNet import RetinaNet
import torch
import logging
import torchvision.transforms as transforms
import os
import sys

from .utlis.detection_helper import create_anchors, process_output, rescale_boxes, cthw2tlbr
from .utlis.nms_WSI import nms_patch


def resource_path(relative_path):
    """Get absolute path to resource (for dev and for PyInstaller onefile mode)"""
    if hasattr(sys, '_MEIPASS'):
        # _MEIPASS is the temp folder where PyInstaller unpacks files
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(""), relative_path)


class MyMitosisDetection:
    def __init__(self, path, config, detect_threshold=0.4, nms_threshold=0.1):

        with open(resource_path(r"retinanet/file/statistics_sdata.pickle"), "rb") as handle:
            statistics = pickle.load(handle)
        tumortypes = config["data"]["value"]["tumortypes"].split(",")
        self.mean = np.array(
            np.mean(np.array([value for key, value in statistics['mean'].items() if tumortypes.__contains__(key)]),
                    axis=(0, 1)), dtype=np.float32)
        self.std = np.array(
            np.mean(np.array([value for key, value in statistics['std'].items() if tumortypes.__contains__(key)]),
                    axis=(0, 1)), dtype=np.float32)

        # Torch downloading weights in exe/packaged app will result in a crash
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w')

        # network parameters
        self.detect_thresh = detect_threshold
        self.nms_threshold = nms_threshold
        encoder = create_body(models.resnet18, True, -2)
        scales = [float(s) for s in config["retinanet"]["value"]["scales"].split(",")]
        ratios = [config["retinanet"]["value"]["ratios"]]
        sizes = [(config["retinanet"]["value"]["sizes"], config["retinanet"]["value"]["sizes"])]
        self.model = RetinaNet(encoder, n_classes=2, n_anchors=len(scales) * len(ratios),
                               sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3)
        # self.path_model = os.path.join(path, "bestmodel.pth")
        self.path_model = path
        self.size = config["data"]["value"]["patch_size"]
        self.batchsize = config["data"]["value"]["batch_size"]

        self.anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    def load_model(self):
        if torch.cuda.is_available():
            print("Model loaded on CUDA")
        else:
            print("Model loaded on CPU")

        self.model.load_state_dict(torch.load(self.path_model, map_location=self.device, weights_only=False)['model'])

        self.model.to(self.device)

        logging.info("Model loaded. Mean: {} ; Std: {}".format(self.mean, self.std))
        return True

    def run_on_tiles(self, tile_batch):  # tile_batch: [B, H, W, 3] or similar
        self.model.eval()
        batch_images = []
        for img in tile_batch:
            cur = img.astype(np.float32) / 255.
            cur = cur.transpose(2, 0, 1)[0:3]  # CHW
            batch_images.append(cur)

        torch_batch = torch.from_numpy(np.stack(batch_images)).to(self.device)
        for p in range(torch_batch.shape[0]):
            torch_batch[p] = transforms.Normalize(self.mean, self.std)(torch_batch[p])

        # -------- Forward pass --------
        with torch.no_grad():
            class_pred_batch, bbox_pred_batch, _ = self.model(torch_batch)


        raw_scores = []
        detections_per_tile = []

        t_sz = torch.tensor([[self.size, self.size]], device=self.device).float()

        for b in range(torch_batch.shape[0]):
            # decode + threshold
            out = process_output(
                class_pred_batch[b],
                bbox_pred_batch[b],
                self.anchors,
                detect_thresh=self.detect_thresh,  # should be <= min(thresholds)
                use_sigmoid=True,
            )
            bbox_pred, scores, preds = out["bbox_pred"], out["scores"], out["preds"]

            if bbox_pred is None:
                detections_per_tile.append(np.zeros((0, 4), dtype=np.float32))
                raw_scores.append([0])
                continue

            # NMS in anchor space
            keep = nms_patch(bbox_pred, scores, self.nms_threshold)
            bbox_pred = bbox_pred[keep]
            scores = scores[keep]  # the conf scores
            preds = preds[keep]


            # raw_scores += scores.detach().cpu().tolist()
            raw_scores.append(scores.detach().cpu().tolist())



            #########################################
            def rescale_tlbr(norm_tlbr: torch.Tensor, t_sz: torch.Tensor) -> torch.Tensor:
                # norm_tlbr: [N,4] in [-1,1], order [top, left, bottom, right] because your cthw2tlbr uses [:2] and [2:]
                # BUT your code later treats first two as (y,x). So keep that convention.
                # t_sz: [[H,W]]
                out = norm_tlbr.clone()
                out[:, 0] = (out[:, 0] + 1) * t_sz[0, 0] / 2  # y1
                out[:, 2] = (out[:, 2] + 1) * t_sz[0, 0] / 2  # y2
                out[:, 1] = (out[:, 1] + 1) * t_sz[0, 1] / 2  # x1
                out[:, 3] = (out[:, 3] + 1) * t_sz[0, 1] / 2  # x2
                return out


            # cthw (norm) -> tlbr (norm) -> tlbr (pixels)
            tlbr_norm = cthw2tlbr(bbox_pred)
            tlbr_px = rescale_tlbr(tlbr_norm, t_sz)  # [y1,x1,y2,x2] in pixels

            # reorder to x1,y1,x2,y2 for drawing
            x1 = tlbr_px[:, 1]
            y1 = tlbr_px[:, 0]
            x2 = tlbr_px[:, 3]
            y2 = tlbr_px[:, 2]

            det = torch.stack([x1, y1, x2, y2], dim=1)

            detections_per_tile.append(det.detach().cpu().numpy())

            # for box in torch.stack([x1, y1, x2, y2], dim=1).detach().cpu().numpy():
            #     convnext_scores.append(self.convnext.get_mitosis_score(tile_batch[b], box))



        return raw_scores, detections_per_tile
