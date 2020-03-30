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

LEARNING_RATE = 0.0001
NUMBER_OF_EPOCH = 200
ITERATIONS = 10
MINI_BATCH = 128
IMG_SHAPE = (720, 1280, 3)

TRAIN_DATA_DIR = 'F:\Python Projects\datasetcreator\images\\train\\'
TEST_DATA_DIR = ""
ANCHORS_RATIO = [(1,1), (0.8, 1.2), (1.2, 0.8)]
ANCHORS_SIZE = [96**2, 128**2, 256**2]

def getGroundTruthBBoxes(train_sample):
    gr = []
    for i in train_sample:
        if isinstance(i, list):
            gr.append(i)

    return gr

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

def getTarget(anchors, gr, img):

    target = []

    for i in anchors:
        for j in gr:
            anch_IOU = bb_intersection_over_union(j, i.getPoints())
            # if anch_IOU > 0.5:
            #     points = i.getPoints()
            #     img = cv2.rectangle(img, (j[0], j[1]), (j[2], j[3]), (0, 255, 0))
            #     img = cv2.rectangle(img, (points[0], points[1]), (points[2], points[3]), (0, 255, 0))
            #     cv2.imshow("test", img)
            #     cv2.waitKey()
            if anch_IOU > 0:
                if anch_IOU < 0.3:
                    i.setCls(0)
                    target.append(i)
                else:
                    i.setCls(1)
                    i.setBbox(calc_reggression(j, i.getPoints()))
                    target.append(i)
    return list(dict.fromkeys(target))

def clearTarget(target):
    cleared_target = {}
    object_target = []
    nonobject_target = []

    for t in range(len(target)):
        if target[t].getCls() == 1:
            object_target.append(target[t])

    for t in range(len(target)):
        if target[t].getCls() == 0:
            nonobject_target.append(target[t])

    # try:
    #     #     target = random.sample(object_target, MINI_BATCH) + random.sample(nonobject_target, MINI_BATCH)
    #     # except:
    #     #     target = object_target + random.sample(nonobject_target, MINI_BATCH)

    try:
        target = object_target[0:MINI_BATCH] + nonobject_target[0:MINI_BATCH]
    except:
        target = object_target + nonobject_target[0:MINI_BATCH]

    for t in target:
        cleared_target[t.getIndex()] = []
    for t in target:
        cleared_target[t.getIndex()].append(t)

    return cleared_target

def model_saving(model, loss):
    path = getcwd()
    for i in listdir(path):
        if isfile(join(path, i)) and 'latest_model_with_' in i:
            #print(i)
            remove(i)

    with open("latest_model_with_" + str(loss) + "_loss", "wb") as f:
        dump(model, f)

# def getTarget(gr, anchors):
#
#     target = []
#
#     for anchor in anchors:
#         anch_IOU = anchor.getIou()
#         if anch_IOU >= 0.5:
#             target.append((1, anchor.getPoints(), calc_reggression(gr, anchor.getPoints())))
#         elif anch_IOU < 0.3 and anch_IOU > 0.0:
#             target.append((0, anchor.getPoints()))
#
#     return target
#
# def getClear_Target(anchors):
#     unclear_anchors = []
#     for i in anchors:
#         if i.getCls() >= 0:
#             unclear_anchors.append(i)
#     return unclear_anchors
#     pass

if __name__ == '__main__':

    vgg = VGG16_model(IMG_SHAPE)

    r = RPN()

    with open("train_data.pickle", 'rb') as f:
        train_data = load(f)
        train_data = np.array(train_data)


    for epoch in range(NUMBER_OF_EPOCH):

        print('--- Epoch %d ---' % (epoch + 1))

        loss = 0

        for iteration in range(ITERATIONS):

            sample = random.choice(train_data)

            img = cv2.imread(TRAIN_DATA_DIR + sample[-1])

            extracted_features = vgg.extract_feature(img)[0]

            gr = getGroundTruthBBoxes(sample)

            # img = cv2.resize(img, (600, 800))

            img, anchors = anch.generate_anchors(img, createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)

            # for g in gr:
            #     img = cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (0, 255, 0))
            # cv2.imshow("test", img)
            # cv2.waitKey()

            pred_anchors = copy.deepcopy(anchors)

            pred_anchors = r.forward_RPN_for_all_anchors(extracted_features, pred_anchors)

            target = getTarget(anchors, gr, img)

            target = clearTarget(target)

            loss += r.getLoss_function(target, pred_anchors, LEARNING_RATE)

        print("Epoch " + str(epoch+1) + " ended with " + str(loss/ITERATIONS) + " loss")

        model_saving(r, loss)







