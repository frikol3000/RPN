from Conv1x1 import Conv1x1
from Conv3x3 import Conv3x3
from softmax import softmax

class RPN:
    def __init__(self):
        self.Conv3x3 = Conv3x3(512)
        self.bboxConv = Conv1x1(512, 36)
        self.clsConv = Conv1x1(512, 18)

    def region_extraction(self, feature_map):

        h, w, _ = feature_map.shape

        for i in range(h-2):
            for j in range(w-2):
                yield feature_map[i:i+3, j:j+3, :]
                #print(i, j)

    def getFeatures(self, region):
        return self.Conv3x3.forward(region)

    def forward_cls(self, features):
        return self.clsConv.forward(features)

    def forward_bbox(self, features):
        return self.bboxConv.forward(features)

    def forward_RPN(self, feature_map, anchors):
        for i, region in enumerate(self.region_extraction(feature_map)):
            cls = (self.forward_cls(self.getFeatures(region)))
            for j, anchor in enumerate(anchors[0+(i*9):9+(i*9)]):
                anchor.setCls(softmax(cls[j:j+2]))
        return  anchors

    def getLoss_function(self, target, proposals):
                return loss







