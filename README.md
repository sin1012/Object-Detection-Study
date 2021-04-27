# Object-Detection-Study
In this repository, I will be studying object detection more in depth.

## Basic I: IOU Calculation
IOU, or intersection over union is an important evaluation metric in object detection.  
A 2D bbox can be represented differently, but I prefer two corner representation. In specific, each bbox can be represented as `[x1, y1, x2, y2]`
A simple example of IOU calculation of 2 bboxes:
```python
def iou(bbox1, bbox2):
  # bbox: [x1, y1, x2, y2]
  bbox1, bbox2 = sorted([bbox1, bbox2], key=lambda x: x[1], reverse=True)
  x1 = max(bbox1[0], bbox2[0])
  y1 = max(bbox1[1], bbox2[1])
  x2 = min(bbox1[2], bbox2[2])
  y2 = min(bbox1[3], bbox2[3])
  intersection = max(x2-x1, 0) * max(y2-y1, 0)
  union = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])) + \
          abs((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
  return intersection/max(union, 1e-8)
```
What if we want to calculate the IOU of multiple bboxes pairs (pred_bbox, gt_bbox). Instead of iterate over them, we can use a vectorized form:
```python
def iou_multiple_bbox(bbox1, bbox2):
  bbox1 = np.array(bbox1)
  bbox2 = np.array(bbox2)
  x1 = np.max([bbox1[:, 0], bbox2[:, 0]], 1)
  y1 = np.max([bbox1[:, 1], bbox2[:, 1]], 1)
  x2 = np.min([bbox1[:, 2], bbox2[:, 2]], 1)
  y2 = np.min([bbox1[:, 3], bbox2[:, 3]], 1)
  intersection = np.clip((x2 - x1), a_min=0, a_max=np.inf) * np.clip((y2-y1), a_min=0, a_max=np.inf)
  union = abs((bbox1[:, 2]-bbox1[:, 0])*(bbox1[:, 3]-bbox1[:, 1])) + \
          abs((bbox2[:, 2]-bbox2[:, 0])*(bbox2[:, 3]-bbox2[:, 1]))
  return intersection / union
```

## Basic II: Non Max Suppression
The non max suppression algorithm works as the follows:
1. we pick the bbox **B** with the highest probability and output it
2. we discard all bboxes that have a high IOU >= threshold with **B**
```python
def nms(bboxes, p_thresh, iou_thresh):
  if isinstance(bboxes, list):
    # prediction is in the format of c, p, bx, by, bh, bw
    # this is our result
    selected_bboxes = []
    # dicard bboxes under the prob threshold
    bboxes = [b for b in bboxes if b[1] >= p_thresh]
    # first sort the bboxes with the higest probablity
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    # while there are bboxes, keep popping the first one
    while bboxes:
      next_bbox = bboxes.pop(0)
      selected_bboxes.append(next_bbox)
      bboxes = [b for b in bboxes if b[0] != next_bbox[0] or 
                iou(b, next_bbox) < iou_thresh]
    return selected_bboxes
  else:
    print('it is not a list!')
```
