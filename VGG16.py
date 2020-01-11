import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image


class VGG16_model:
    def __init__(self, inp_shape):
        self.__inp_shape = inp_shape
        self.__model = VGG16(weights='imagenet', include_top=False, input_shape=self.__inp_shape)
        self.__model.summary()

    def extract_feature(self, img_raw):
        img_data = image.img_to_array(img_raw)
        img_data = np.expand_dims(img_data, axis=0)
        vgg16_feature = self.__model.predict(img_data)

        return vgg16_feature

    def getModel(self):
        return self.__model

    def setMode(self, model):
        self.__model = model