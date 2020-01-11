import cv2
import keyboard
import numpy as np
from PIL import ImageGrab

w = 0
h = 0

key_rightpress = False
key_leftpress = False
key_uptpress = False
key_downpress = False
key_right = 'right'
key_left = 'left'
key_up = 'up'
key_down = 'down'

while(True):
    im = np.array(ImageGrab.grab(bbox=(0 + w, 0 + h, 500 + w, 500 + h)))
    cv2.imshow("Dataset_creator", im)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    if key_rightpress and not keyboard.is_pressed(key_right):
        w += 40
        print("w = {0}".format(w))
        # beak out of while loop?
        key_rightpress = False
    elif keyboard.is_pressed(key_right) and not key_rightpress:
        key_rightpress = True
    if key_leftpress and not keyboard.is_pressed(key_left):
        w -= 40
        print("w = {0}".format(w))
        # beak out of while loop?
        key_leftpress = False
    elif keyboard.is_pressed(key_left) and not key_leftpress:
        key_leftpress = True
    if key_uptpress and not keyboard.is_pressed(key_up):
        h -= 40
        print("h = {0}".format(h))
        # beak out of while loop?
        key_uptpress = False
    elif keyboard.is_pressed(key_up) and not key_uptpress:
        key_uptpress = True
    if key_downpress and not keyboard.is_pressed(key_down):
        h += 40
        print("h = {0}".format(h))
        # beak out of while loop?
        key_downpress = False
    elif keyboard.is_pressed(key_down) and not key_downpress:
        key_downpress = True
