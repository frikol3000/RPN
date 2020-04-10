import urllib.request
import cv2
import numpy as np
from VGG16 import VGG16_model
from os import listdir, getcwd, remove
from os.path import isfile, join
import cv2
import generate_anchors as anch
from VGG16 import VGG16_model
from pickle import load
import numpy as np
from nms import non_max_suppression_slow
from create_dataset_RPN import createPoints

IMG_SHAPE = (480, 640, 3)
ANCHORS_RATIO = [(1, 1), (0.7, 1.3), (1.3, 0.7)]
ANCHORS_SIZE = [96**2, 156**2, 208**2]
url = 'http://192.168.0.100:8080/shot.jpg'
vgg = VGG16_model(IMG_SHAPE)

def getModelName():
    path = getcwd()
    for i in listdir(path):
        if isfile(join(path, i)) and 'latest_model_with_' in i:
            return i

with open(getModelName(), 'rb') as R:
    r = load(R)
    print("Loaded last model " + str(getModelName()))

def get_Bboxes(img):

    #t1 = time.time()

    extracted_features = vgg.extract_feature(img)[0]

    pred_anchors, anchors = anch.generate_anchors(createPoints(img)[0], ANCHORS_RATIO, ANCHORS_SIZE)

    pred_anchors = r.forward_RPN_for_all_anchors(extracted_features, pred_anchors)

    boxes = non_max_suppression_slow(pred_anchors, 0.1)

    #print(time.time() - t1)

    return boxes

while True:
    imageResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imageResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    #cv2.imwrite(dir + str(counter) + ".jpg", img)

    for i in get_Bboxes(img):
        img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0))

    #put the image on screen
    cv2.imshow('IPWebcam', img)

    # To give the processor some less stress

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()