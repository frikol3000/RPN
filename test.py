import cv2
import csv
from os import listdir, getcwd, remove
from os.path import isfile, join
import numpy as np
import pickle
import random
from VGG16 import VGG16_model
import generate_anchors as anch
import RPN

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


with open("train_data.pickle", 'rb') as f:
    train_data = pickle.load(f)
    train_data = np.array(train_data)

path = getcwd()
for i in listdir(path):
    if isfile(join(path, i)) and 'latest_model_with_' in i:
        with open(i, 'rb') as f:
            r = pickle.load(f)

# r = RPN.RPN()
vgg = VGG16_model((720, 1280, 3))

sample = random.choice(train_data)

#img = cv2.imread(TRAIN_DATA_DIR + sample[-1])
img = cv2.imread("CARDS_LIVINGROOM_S_H_frame_1716.jpg")

extracted_features = vgg.extract_feature(img)[0]

img, anchors = anch.generate_anchors(img, createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)

for i in r.forward_RPN2(extracted_features, anchors):
    if i.getCls() > 0.9999999:
        points = i.getPoints()
        img = cv2.rectangle(img, (points[0], points[1]), (points[2], points[3]), (0, 255, 0))
cv2.imshow("test", img)
cv2.waitKey()



# import Conv1x1
# import Conv3x3
#
# c = Conv1x1.Conv1x1(512, 18)
# c3x3 = Conv3x3.Conv3x3(512)
#
# arr = np.random.uniform(0.0, 1.0, size=(20, 20, 512))
#
# # print(c3x3.forward(arr[0:3, 0:3]).shape)
# # print(c.forward(c3x3.forward(arr[0:3, 0:3])).shape)
#
# print(c.filters[0].shape)






