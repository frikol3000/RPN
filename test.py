import cv2
import csv
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pickle

LABEL_DIR = 'F:\Python Projects\datasetcreator\images\\train\\train_data.csv'

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

class Something:
    def __init__(self, num):
        self.num = num

    def change(self, num):
        self.num += num

s = Something(5)

print(s.num)

s.change(6)

print(s.num)






