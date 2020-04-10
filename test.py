from os import listdir, getcwd, remove
from os.path import isfile, join
import cv2
import generate_anchors as anch
from utils import bb_intersection_over_union
from utils import calc_reggression
from RPN import RPN
from VGG16 import VGG16_model
from pickle import load, dump
import numpy as np
import copy
from utils import CrossEntropy
from utils import softmax
import random

TRAIN_DATA_DIR = 'F:\Python Projects\datasetcreator\images\\train\\'
MUTATION_RATE = 0.02
NUMBER_OF_EPOCH = 200
ITERATIONS = 10
MINI_BATCH = 128
IMG_SHAPE = (720, 1280, 3)

ANCHORS_RATIO = [(1,1), (0.8, 1.2), (1.2, 0.8)]
ANCHORS_SIZE = [96**2, 128**2, 256**2]

def show_pic_with_gr(dir_to_csv):
    with open(dir_to_csv) as cvsfile:
        x = csv.reader(cvsfile, delimiter=',')
        gr = []
        for i in x:
            if i != []:
                i[4:8] = [int(x) for x in i[4:8]]
                gr.append(i[4:8])

    img = cv2.imread(dir_to_csv[:-3] + "jpg")

    for point in gr:
        max_x = point[0] if point[0] > point[2] else point[2]
        max_y = point[1] if point[1] > point[3] else point[3]
        min_x = point[0] if point[0] < point[2] else point[2]
        min_y = point[1] if point[1] < point[3] else point[3]

        print(max_x, max_y)
        img = cv2.rectangle(img, (max_x, max_y), (min_x, min_y), (0, 255, 0))
    cv2.imshow("2", img)
    cv2.waitKey(0)

def createPoints(img):

    start_x = 48
    start_y = 48
    points = []
    points_feature_map = []
    h, w, _ = img.shape

    h = h//32
    w = w//32

    for i in range(h-2):
        for j in range(w-2):
            points.append((start_x, start_y))
            points_feature_map.append((j, i))
            start_x += 32
        start_y += 32
        start_x = 48

    return points, points_feature_map


# with open("train_data.pickle", 'rb') as f:
#     train_data = pickle.load(f)
#     train_data = np.array(train_data)
#
# path = getcwd()
# for i in listdir(path):
#     if isfile(join(path, i)) and 'latest_model_with_' in i:
#         with open(i, 'rb') as f:
#             r = pickle.load(f)
#
# # r = RPN.RPN()
# vgg = VGG16_model((720, 1280, 3))
#
# sample = random.choice(train_data)
#
# #img = cv2.imread(TRAIN_DATA_DIR + sample[-1])
# img = cv2.imread("CARDS_LIVINGROOM_S_H_frame_1716.jpg")
#
# extracted_features = vgg.extract_feature(img)[0]
#
# img, anchors = anch.generate_anchors(img, createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)
#
# for i in r.forward_RPN2(extracted_features, anchors):
#     if i.getCls() > 0.9999999:
#         points = i.getPoints()
#         img = cv2.rectangle(img, (points[0], points[1]), (points[2], points[3]), (0, 255, 0))
# cv2.imshow("test", img)
# cv2.waitKey()

#
#
# from Conv3x3 import Conv3x3
#
# vgg = VGG16_model((720, 1280, 3))
#
# img = cv2.imread("CARDS_LIVINGROOM_S_H_frame_1716.jpg")
#
# features = vgg.extract_feature(img)[0]
#
# c = Conv3x3(2)
#
# features_conv = features[19:22, 37:40, 0:2]
# features_conv[0, 0, 1] = 1
#
# print(features_conv.shape)
# print(c.filters[:, :, :])
# print(c.forward(features_conv))

# a = np.array(([1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]))
# a = a.reshape((3,3,2))
#
# print(a)

from utils import CrossEntropy
# def softmax(x):
#     x = x[0][0]
#     temp = []
#     for i in x:
#         temp.append(np.exp(i)/np.sum(np.exp(x), axis=0))
#     return temp
#
# t1 = np.random.uniform(size=(3,3,1))
# filt = np.random.uniform(size=(3,3,1))
# filt1x1 = np.random.uniform(size=(1, 1, 2))
#
#
# for i in range(10000):
#     d_l_conv1x1 = np.zeros((1, 1, 2))
#     d_l_conv3x3 = np.zeros((3, 3, 1))
#     for j, t in enumerate([1, 0]):
#         conv3x3_out = (np.sum(t1 * filt, axis=(0,1)).reshape((1, 1, 1)))
#         #print(conv3x3_out)
#         # print(filt1x1)
#         # print(np.multiply(conv3x3_out, filt1x1))
#         soft = softmax(np.multiply(conv3x3_out, filt1x1))
#         # print(filt)
#         # print(soft)
#         # print(np.multiply(filt1x1[:,:,0], t1))
#         print(j, soft[j], t)
#         d_l_conv1x1[:, :, j] = (-1) * np.multiply((t - soft[j]), conv3x3_out).reshape((1,1))
#         d_l_conv3x3 += (-1) * (t - soft[j]) * np.multiply(filt1x1[:,:,j], t1)
#         #print(d_l_conv1x1)
#         print(d_l_conv3x3)
#         print(soft)
#     #filt1x1 -= 0.01 * d_l_conv1x1
#     filt -= 0.01 * d_l_conv3x3
#
# def getTarget(anchors, gr):
#
#     target = []
#
#     for i in anchors:
#         for j in gr:
#             anch_IOU = bb_intersection_over_union(j, i.getPoints())
#             if anch_IOU > 0:
#                 if anch_IOU < 0.3:
#                     i.setCls(0)
#                     target.append(i)
#                 else:
#                     i.setCls(1)
#                     i.setBbox(calc_reggression(j, i.getPoints()))
#                     target.append(i)
#     return list(dict.fromkeys(target))
#
# def clearTarget(target):
#     cleared_target = {}
#     object_target = []
#     nonobject_target = []
#
#     for t in range(len(target)):
#         if target[t].getCls() == 1:
#             object_target.append(target[t])
#
#     for t in range(len(target)):
#         if target[t].getCls() == 0:
#             nonobject_target.append(target[t])
#
#     # try:
#     #     target = random.sample(object_target, MINI_BATCH) + random.sample(nonobject_target, MINI_BATCH)
#     # except:
#     #     target = object_target + random.sample(nonobject_target, MINI_BATCH)
#
#     try:
#         target = object_target[0:MINI_BATCH] + nonobject_target[0:MINI_BATCH]
#     except:
#         target = object_target + nonobject_target[0:MINI_BATCH]
#
#     for t in target:
#         cleared_target[t.getIndex()] = []
#     for t in target:
#         cleared_target[t.getIndex()].append(t)
#
#     return cleared_target
#
# def getModelName():
#     path = getcwd()
#     for i in listdir(path):
#         if isfile(join(path, i)) and 'latest_model_with_' in i:
#             return i
#
# def getGroundTruthBBoxes(train_sample):
#     gr = []
#     for i in train_sample:
#         if isinstance(i, list):
#             gr.append(i)
#
#     return gr
#
# r = RPN()
#
# vgg = VGG16_model(IMG_SHAPE)
#
# with open("train_data.pickle", 'rb') as f:
#     train_data = load(f)
#     train_data = np.array(train_data)
#
# sample = random.choice(train_data)
#
# img = cv2.imread(TRAIN_DATA_DIR + sample[-1])
#
# extracted_features = vgg.extract_feature(img)[0]
#
# for i in range(250):
#     loss = 0
#
#     gr = getGroundTruthBBoxes(sample)
#
#     # img = cv2.resize(img, (600, 800))
#
#     img, anchors = anch.generate_anchors(img, createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)
#
#     pred_anchors = copy.deepcopy(anchors)
#
#     pred_anchors = r.forward_RPN_for_all_anchors(extracted_features, pred_anchors)
#
#     target = getTarget(anchors, gr)
#
#     target = clearTarget(target)
#
#     loss += r.getLoss_function(target, pred_anchors, 0.001)
#
#     print(loss)
#
# img, anchors = anch.generate_anchors(img, createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)
#
# pred_anchors = copy.deepcopy(anchors)
#
# pred_anchors = r.forward_RPN_for_all_anchors(extracted_features, pred_anchors)
#
# gr = getGroundTruthBBoxes(sample)
#
# for g in gr:
#     img = cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (255, 0, 0))
#
# proposal_list_and_cls = []
#
# for i in pred_anchors:
#     proposal_list_and_cls.append((i, i.getCls()))
#
# proposal_list_and_cls = sorted(proposal_list_and_cls, key=lambda tup: tup[1])
# for i in range(300):
#     temp = proposal_list_and_cls[i][0]
#     img = cv2.rectangle(img, (temp.getPoints()[0], temp.getPoints()[1]), (temp.getPoints()[2], temp.getPoints()[3]), (0, 255, 0))
# cv2.imshow("test", img)
# cv2.waitKey()

# import the necessary packages
import numpy as np
#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # initialize the list of picked indexes
    pick = []

    boxes = np.array([
	(12, 84, 140, 212),
	(24, 84, 152, 212),
	(36, 84, 164, 212),
	(12, 96, 140, 224),
	(24, 96, 152, 224),
	(24, 108, 152, 236)])

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        print(i)
        pick.append(i)
        print(pick)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in list(range(0, last)):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)
            # return only the bounding boxes that were picked
    return boxes[pick]

non_max_suppression_slow([], 0.3)




