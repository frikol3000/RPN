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
import time
from nms import non_max_suppression_slow

LEARNING_RATE = 0.0001
NUMBER_OF_EPOCH = 200
ITERATIONS = 10
BATCH = 500
MINI_BATCH = 90
IMG_SHAPE = (480, 640, 3)

TRAIN_DATA_DIR = 'F:\Python Projects\Faster - RCNN\images\\train\\'
TEST_DATA_DIR = ""
ANCHORS_RATIO = [(1, 1), (0.7, 1.3), (1.3, 0.7)]
ANCHORS_SIZE = [96**2, 128**2, 156**2]

vgg = VGG16_model(IMG_SHAPE)

def getModelName():
    path = getcwd()
    for i in listdir(path):
        if isfile(join(path, i)) and 'latest_model_with_' in i:
            return i

def getGroundTruthBBoxes(train_sample):
    gr = []
    for i in train_sample["bboxes"]:
        gr.append(i[1])

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

def getTarget(anchors, gr):

    target = []

    for i in anchors:
        for j in gr:
            anch_IOU = bb_intersection_over_union(j, i.getPoints())
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

    if len(object_target) < MINI_BATCH:
        object_target = object_target + object_target

    #print(len(object_target))

    try:
        target = random.sample(object_target, MINI_BATCH) + random.sample(nonobject_target, MINI_BATCH)
    except:
        target = object_target + nonobject_target

    # try:
    #     target = object_target[0:MINI_BATCH] + nonobject_target[0:MINI_BATCH]
    # except:
    #     target = object_target + nonobject_target[0:MINI_BATCH]

    for t in target:
        cleared_target[t.getIndex()] = []
    for t in target:
        cleared_target[t.getIndex()].append(t)

    return cleared_target

def model_saving(model, loss):
    path = getcwd()
    for i in listdir(path):
        if isfile(join(path, i)) and 'latest_model_with_' in i:
            remove(i)

    with open("latest_model_with_" + str(loss) + "_loss.pickle", "wb") as f:
        dump(model, f)

with open(getModelName(), 'rb') as R:
    r = load(R)
    print("Loaded last model " + str(getModelName()))


if __name__ == '__main__':
    vgg = VGG16_model(IMG_SHAPE)

    while (1):
        ch = input("Load - l, Create - c\n")
        if ch == "l":
            with open(getModelName(), 'rb') as R:
                r = load(R)
                print("Loaded last model " + str(getModelName()))
                break
        elif ch == "c":
            r = RPN()
            print("New model created")
            break
        else:
            print("Incorrect input. Please try again")

    with open("train_set.pickle", 'rb') as f:
        train_data = load(f)

    for epoch in range(NUMBER_OF_EPOCH):

        t1 = time.time()

        loss = 0

        sample = random.sample(train_data, BATCH)

        print('--- Epoch %d ---' % (epoch + 1))

        for single_sample in sample:

            img = cv2.imread(single_sample['dir'])

            extracted_features = vgg.extract_feature(img)[0]

            gr = getGroundTruthBBoxes(single_sample)

            # for g in gr:
            #     img = cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (0, 255, 0))
            # cv2.imshow("test", img)
            # cv2.waitKey()

            # img = cv2.resize(img, (600, 800))

            pred_anchors, anchors = anch.generate_anchors(createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)

            pred_anchors = r.forward_RPN_for_all_anchors(extracted_features, pred_anchors)

            print(len(anchors))

            # for i in pred_anchors:
            #    if i.getCls() > 0.9999:
            #     img = cv2.rectangle(img, (i.getPoints()[0], i.getPoints()[1]), (i.getPoints()[2], i.getPoints()[3]), (0, 255, 0))
            # cv2.imshow("test", img)
            # cv2.waitKey()

            target = getTarget(anchors, gr)

            target = clearTarget(target)

            # counter = 0
            #
            # for t in target:
            #     for anchor in target[t]:
            #         if anchor.getCls() == 1:
            #             counter += 1
            #             img = cv2.rectangle(img, (anchor.getPoints()[0], anchor.getPoints()[1]), (anchor.getPoints()[2], anchor.getPoints()[3]), (0, 255, 0))
            #
            # print(counter)
            # cv2.imshow("Test", img)
            # cv2.waitKey()

            # for t in target:
            #     for anchor in target[t]:
            #         if anchor.getCls() != -1:
            #             print(anchor.getCls())

            # dict = {1: [886, 585, 1001, 662], 0:[880, 560, 1008, 688]}
            #
            # d_l = np.zeros((1, 1, 512, 18))
            # d_l_conv3x3 = np.zeros((3, 3, 512))
            #
            # for t in dict:
            #     for proposal in pred_anchors:
            #         if proposal.getPoints() == dict[t]:
            #             region_featuremap = proposal.getFeature()[1]
            #             feature = proposal.getFeature()[0]
            #             loss += (CrossEntropy(proposal.getCls(), t))
            #             d_l[:, :, :, proposal.getSetNum() * 2] -= (1/2) * (t - softmax(proposal.getX1_X2())[0]) * feature
            #             d_l[:, :, :, (proposal.getSetNum() * 2) + 1] -= (1/2) * ((1-t) - softmax(proposal.getX1_X2())[1]) * feature
            #             d_l_conv3x3 -= (1/2) * (t - softmax(proposal.getX1_X2())[0]) * np.multiply(
            #             r.clsConv.filters[:, :, :, proposal.getSetNum() * 2],
            #             region_featuremap)
            #             d_l_conv3x3 -= (1/2) * ((1 - t) - softmax(proposal.getX1_X2())[1]) * np.multiply(
            #                 r.clsConv.filters[:, :, :, (proposal.getSetNum() * 2) + 1],
            #                 region_featuremap)
            #             print(softmax(proposal.getX1_X2()), t, (proposal.getX1_X2()))
            #             #print(proposal.getCls(), t)
            #
            # #print(d_l_conv3x3)
            # #print(d_l[0][0][100:110, :])
            # #print(d_l)
            # r.clsConv.backpropagate(d_l, LEARNING_RATE)
            # r.Conv3x3.backpropagate(d_l_conv3x3, LEARNING_RATE)
            # print(loss)

            # d_l_cls = np.zeros((1, 1, 512, 18))
            # d_l_conv3x3 = np.zeros((3, 3, 512))
            #
            # for t in target:
            #     for anchor in target[t]:
            #         for proposal in pred_anchors:
            #             if proposal.getPoints() == anchor.getPoints():
            #                 region_featuremap = proposal.getFeature()[1]
            #                 feature = proposal.getFeature()[0]
            #                 loss += (1/MINI_BATCH) * (CrossEntropy(proposal.getCls(), anchor.getCls()))
            #                 d_l_cls[:, :, :, proposal.getSetNum() * 2] -= (1/MINI_BATCH) * (anchor.getCls() - softmax(proposal.getX1_X2())[
            #                     0]) * feature
            #                 d_l_cls[:, :, :, (proposal.getSetNum() * 2) + 1] -= (1/MINI_BATCH) * ((1 - anchor.getCls()) - softmax(proposal.getX1_X2())[
            #                     1]) * feature
            #                 d_l_conv3x3 -= (1/MINI_BATCH) *  ((1 - anchor.getCls()) - softmax(proposal.getX1_X2())[1]) * np.multiply(
            #                     r.clsConv.filters[:, :, :, (proposal.getSetNum() * 2) + 1],
            #                     region_featuremap)
            #                 d_l_conv3x3 -= (1/MINI_BATCH) *  (anchor.getCls() - softmax(proposal.getX1_X2())[0]) * np.multiply(
            #                     r.clsConv.filters[:, :, :, (proposal.getSetNum() * 2)],
            #                     region_featuremap)
            #                 print(softmax(proposal.getX1_X2()), anchor.getCls(), (proposal.getX1_X2()))
            #                 # print(proposal.getCls(), anchor.getCls())
            #
            # #print(d_l_conv3x3)
            # #print(d_l_cls[0][0][100:110, 5])
            # r.clsConv.backpropagate(d_l_cls, LEARNING_RATE)
            # r.Conv3x3.backpropagate(d_l_conv3x3, LEARNING_RATE)
            #
            # print(loss)

            loss += r.getLoss_function(target, pred_anchors, LEARNING_RATE)

        print("Epoch " + str(epoch + 1) + " ended with " + str(
            loss / BATCH) + " average loss and took %.2f seconds" % (time.time() - t1))
        model_saving(r, loss / BATCH)













