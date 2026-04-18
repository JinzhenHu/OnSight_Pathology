import torch
from .detection_helper import IoU_values


def nms_patch(boxes, scores, thresh=0.5):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    # to_keep, indexes = [], torch.LongTensor(np.arange(len(scores)))
    to_keep, indexes = [], torch.arange(len(scores), device=scores.device)

    while len(scores) > 0:
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals <= thresh
        if len(mask_keep.nonzero()) == 0: break
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    return torch.LongTensor(to_keep)
