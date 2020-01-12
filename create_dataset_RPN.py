import re
from os import listdir
from os.path import isfile, join
import cv2
import generate_anchors as anch
from utils import bb_intersection_over_union
from utils import calc_reggression
from RPN import RPN
from VGG16 import VGG16_model

train_data = 100
img = []
img_anchors = {}

onlyfiles = [f for f in listdir("F:\курс\datasetcreator\dataset") if isfile(join("F:\курс\datasetcreator\dataset", f))]

def getGroundTruthBBoxes(str):

    gr = []
    pattern = r"(\d+, \d+, \d+, \d+)"

    str_gr = re.findall(pattern, str)
    for i in str_gr:
        o = i.split(", ")
        o = [int(x) for x in o]
        for count, element in enumerate(o, 1):  # Start counting from 1
            if count % 4 == 0:
                gr.append(o)

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

def setIOU(anchors, gr):
    for i in anchors:
        for j in gr:
            if bb_intersection_over_union(j, i.getPoints()) > 0:
                i.setIou(bb_intersection_over_union(j, i.getPoints()))

def getTarget(gr, anchors):

    target = []

    for anchor in anchors:
        if anchor.getIou() >= 0.5:
            target.append((1, anchor.getPoints(), calc_reggression(gr, anchor.getPoints())))
        elif anchor.getIou() < 0.3 and anchor.getIou() > 0.0:
            target.append((0, anchor.getPoints()))

    return target


for i in onlyfiles:

    temp = cv2.imread("dataset\\" + i)

    gr = getGroundTruthBBoxes(i)

    temp = cv2.resize(temp, (800, 640))
    temp, anchors = anch.generate_anchors(temp, createPoints(temp)[0], [(1,1), (0.8, 1.2), (1.2, 0.8)], [32**2, 48**2, 64**2])

    # temp = cv2.rectangle(temp, (gr[0][0], gr[0][1]), (gr[0][2], gr[0][3]), (0, 255, 0))
    # cv2.imshow("2", temp)
    # cv2.waitKey()

    setIOU(anchors, gr)

    vgg = VGG16_model(temp.shape)
    r = RPN()
    new_anchors = r.forward_RPN(vgg.extract_feature(temp)[0], anchors)

    target = getTarget(gr[0], anchors)

    print(r.getLoss_function(target, new_anchors))



