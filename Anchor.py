class Anchor:
    def __init__(self, cls, bbox, iou_rate, points, index, num_in_set):
        self.__cls = cls
        self.__bbox = bbox
        self.__iou_rate = iou_rate
        self.__points = points
        self.__feature = None
        self.__index = index
        self.__num_in_set = num_in_set
        self.__x1_x2 = ()

    def getCls(self):
        return self.__cls

    def setCls(self, cls):
        self.__cls = cls

    def getBbox(self):
        return self.__bbox

    def setBbox(self, bbox):
        self.__bbox = bbox

    def getIou(self):
        return self.__iou_rate

    def setIou(self, iou_rate):
        self.__iou_rate = iou_rate

    def getPoints(self):
        return self.__points

    def setPoints(self, points):
        self.__points = points

    def getFeature(self):
        return self.__feature

    def setFeature(self, feature):
        self.__feature = feature

    def getIndex(self):
        return self.__index

    def setIndex(self, index):
        self.__index = index

    def getSetNum(self):
        return self.__num_in_set

    def setSetNum(self, num_in_set):
        self.__num_in_set = num_in_set

    def setX1_X2(self, x1_x2):
        self.__x1_x2 = x1_x2

    def getX1_X2(self):
        return self.__x1_x2