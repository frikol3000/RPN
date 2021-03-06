import cv2

from Anchor import Anchor


def generate_anchors(img, points, ratio, size):
    anchors = []
    for i in points:
        center_x, center_y = i
        for k in size:
            for j in ratio:
                # if int((center_x - (pow(k, 0.5) / 2) * (j[0]))) >= 0 and int((center_y - (pow(k, 0.5) / 2) * (j[1]))) >= 0 and int(
                #         (center_x + (pow(k, 0.5) / 2) * (j[0]))) <= img.shape[1] and int((center_y + (pow(k, 0.5) / 2) * (j[1]))) <= img.shape[0]:
                x1, y1 = int((center_x + (pow(k, 0.5) / 2) * (j[0]))), int((center_y + (pow(k, 0.5) / 2) * (j[1])))
                x2, y2 = int((center_x - (pow(k, 0.5) / 2) * (j[0]))), int((center_y - (pow(k, 0.5) / 2) * (j[1])))
                # img = cv2.rectangle(img, (x2, y2), (x1, y1), (0, 0, 5))
                anchors.append(Anchor(-1, [], 0, [x2, y2, x1, y1], [center_x//16, center_y//16]))
    return img, anchors

