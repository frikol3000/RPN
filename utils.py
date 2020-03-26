from numpy import log
import numpy as np

def CrossEntropy(yHat, y):
    if yHat == 1:
        yHat = 0.9
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    #print(boxAArea)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #print(boxBArea)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def softmax(x):
    try:
        return np.exp(x[0]) / np.sum(np.exp(x), axis=0)
    except:
        return x[0] / np.sum(x, axis=0)

def mean_squared_diff(yHat, y):
    yHat = np.array(yHat)
    y = np.array(y)
    return np.sum(np.square(yHat-y))

def calc_reggression(bbox, anchor):
    x, y, h, w = (bbox[0] - bbox[2])/2, (bbox[1] - bbox[3])/2, np.abs(bbox[1] - (bbox[1] - bbox[3])/2), np.abs(bbox[0] - (bbox[0] - bbox[2])/2)
    xA, yA, hA, wA = (anchor[0] - anchor[2]) / 2, (anchor[1] - anchor[3]) / 2, np.abs(anchor[1] - (anchor[1] - anchor[3]) / 2), np.abs(anchor[0] - (anchor[0] - anchor[2]) / 2)

    tx = (x-xA)/wA
    ty = (y-yA)/hA
    th = np.log(h/hA)
    tw = np.log(w/wA)

    return [tx, ty, th, tw]
