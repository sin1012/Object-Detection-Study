import torch 

def iou_2d(pred_bbox, gt_bbox, box_format='mid'):

    if box_format == "mid":
        pred_x1 = pred_bbox[..., 0] - pred_bbox[..., 2] / 2
        pred_y1 = pred_bbox[..., 1] - pred_bbox[..., 3] / 2
        pred_x2 = pred_bbox[..., 0] + pred_bbox[..., 2] / 2
        pred_y2 = pred_bbox[..., 1] + pred_bbox[..., 3] / 2
        gt_x1 = gt_bbox[..., 0] - gt_bbox[..., 2] / 2
        gt_y1 = gt_bbox[..., 1] - gt_bbox[..., 3] / 2
        gt_x2 = gt_bbox[..., 0] + gt_bbox[..., 2] / 2
        gt_y2 = gt_bbox[..., 1] + gt_bbox[..., 3] / 2
    
    if box_format == 'corner':
        pred_x1 = pred_bbox[..., 0]
        pred_y1 = pred_bbox[..., 1]
        pred_x2 = pred_bbox[..., 2]
        pred_y2 = pred_bbox[..., 3]
        gt_x1 = gt_bbox[..., 0]
        gt_y1 = gt_bbox[..., 1]
        gt_x2 = gt_bbox[..., 2]
        gt_y2 = gt_bbox[..., 3]
    
    x1 = torch.max(pred_x1, gt_x1)
    y1 = torch.max(pred_y1, gt_y1)
    x2 = torch.min(pred_x2, gt_x2)
    y2 = torch.min(pred_y2, gt_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # if no intersect, iou=0
    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    gt_area = abs((gt_x2 - gt_x1) * (gt_y2 - gt_y1))
    union_ = (pred_area + gt_area - intersection)
    if union_ == 0:
        union_ == 1e-6 # make sure we don't divide by 0
    return intersection / union_
