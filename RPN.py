from Conv1x1 import Conv1x1
from Conv3x3 import Conv3x3
from utils import softmax
from utils import CrossEntropy
from utils import mean_squared_diff
from utils import calc_reggression

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

    def forward_RPN(self, feature_map, index, anchors):
        for i, region in enumerate(self.region_extraction(feature_map)):
            if i == index/9:
                feature = self.getFeatures(region)
                cls = (self.forward_cls(feature))
                bbox = (self.forward_bbox(feature))
                for j, anchor in enumerate(anchors[0+(i*9):9+(i*9)]):
                    anchor.setCls(softmax(cls[j:j+2]))
                    anchor.setBbox(bbox[j:j+4])
                return anchors[0+(i*9):9+(i*9)]

    def getLoss_function(self, target, proposals, feachures):

        loss = 0

        for t in target:
            for p in self.forward_RPN(feachures, t, proposals):
                for anch in target[t]:
                    if anch.getPoints() == p.getPoints():
                        if anch.getCls() == 1:
                            loss += (1/256) * CrossEntropy(p.getCls(), anch.getCls()) + 10 * (1/6840) * mean_squared_diff(anch.getBbox(), calc_reggression(p.getBbox(), p.getPoints()))
                        else:
                            loss += (1/256) * CrossEntropy(p.getCls(), anch.getCls())
        return loss

    def mutate(self, mutate_rate):
        self.Conv3x3.mutate(mutate_rate)
        self.clsConv.mutate(mutate_rate)
        self.bboxConv.mutate(mutate_rate)
        pass







