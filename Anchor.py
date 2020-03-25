class Anchor:
    def __init__(self, cls, bbox, iou_rate, points, feature_points, index):
        self.__cls = cls
        self.__bbox = bbox
        self.__iou_rate = iou_rate
        self.__points = points
        self.__feature_points = feature_points
        self.__index = index

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

    def getFeaturePoints(self):
        return self.__feature_points

    def setFeaturePoints(self, feature_points):
        self.__feature_points = feature_points

    def getIndex(self):
        return self.__index

    def setIndex(self, index):
        self.__index = index