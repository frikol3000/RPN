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

    def forward_RPN(self, feature_map, anchors):
        for i, region in enumerate(self.region_extraction(feature_map)):
            feature = self.getFeatures(region)
            cls = (self.forward_cls(feature))
            bbox = (self.forward_bbox(feature))
            for j, anchor in enumerate(anchors[0+(i*9):9+(i*9)]):
                anchor.setCls(softmax(cls[j:j+2]))
                anchor.setBbox(bbox[j:j+4])
        return  anchors

    def getLoss_function(self, target, proposals):

        loss = 0

        for t in target:
            for proposal in proposals:
                if t[1] == proposal.getPoints():
                    if t[0] == 1:
                        loss += 0.5 * CrossEntropy(proposal.getCls(), t[0]) + 0.5 * mean_squared_diff(t[2], calc_reggression(proposal.getBbox(), proposal.getPoints()))
                    else:
                        loss += 0.5 * CrossEntropy(proposal.getCls(), t[0])

        return loss

    def mutate(self, mutate_rate):
        pass







