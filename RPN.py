from Conv1x1 import Conv1x1
from Conv3x3 import Conv3x3
from utils import softmax
from utils import CrossEntropy
from utils import mean_squared_diff
from utils import calc_reggression
import numpy as np

class RPN:
    def __init__(self):
        self.Conv3x3 = Conv3x3(512)
        self.bboxConv = Conv1x1(512, 36)
        self.clsConv = Conv1x1(512, 18)

    def getRegion(self, conv_feature):

        h, w, _ = conv_feature.shape

        count = -1

        for i in range(h):
            for j in range(w):
                count += 1
                conv_feature1x1 = conv_feature[i, j, :].reshape((1,1,512))
                yield count, i, j, conv_feature1x1

    def getFeatures(self, region):
        return self.Conv3x3.forward(region)

    def forward_cls(self, features):
        return self.clsConv.forward(features)

    def forward_bbox(self, features):
        return self.bboxConv.forward(features)

    # def forward_RPN(self, feature_map, index, anchors):
    #     for i, region in enumerate(self.region_extraction(feature_map)):
    #         if i == index/9:
    #             feature = self.getFeatures(region)
    #             cls = (self.forward_cls(feature))
    #             bbox = (self.forward_bbox(feature))
    #             for j, anchor in enumerate(anchors[0+(i*9):9+(i*9)]):
    #                 anchor.setCls(softmax(cls[j:j+2]))
    #                 anchor.setBbox(bbox[j:j+4])
    #             return anchors[0+(i*9):9+(i*9)]

    def forward_RPN_for_all_anchors(self, feature_map, anchors):
        feature = self.getFeatures(feature_map)
        for count, i, j, conv_region1x1 in self.getRegion(feature):
            cls = (self.forward_cls(conv_region1x1))
            bbox = (self.forward_bbox(conv_region1x1))
            for k, anch in enumerate(anchors[count*9:(count+1)*9]):
                anch.setX1_X2(cls[k*2:(k+1)*2])
                anch.setCls(softmax(cls[k*2:(k+1)*2])[0])
                anch.setBbox(bbox[k*4:(k+1)*4])
                anch.setFeature(conv_region1x1)
        return anchors

    def getLoss_function(self, target, proposals, learn_rate):

        loss = 0

        for t in target:
            for anchor in target[t]:
                for proposal in proposals:
                    if proposal.getPoints() == anchor.getPoints():
                        if anchor.getCls() == 1:
                            loss += (1/256) * (CrossEntropy(proposal.getCls(), anchor.getCls()))
                            d_l = (softmax(proposal.getX1_X2())[0] - anchor.getCls()) * proposal.getFeature()
                            self.clsConv.backpropagate(d_l, learn_rate, proposal.getSetNum() * 2)
                        else:
                            loss += (1 / 256) * (CrossEntropy(proposal.getCls(), anchor.getCls()))
                            d_l = (anchor.getCls() - softmax(proposal.getX1_X2())[0]) * proposal.getFeature()
                            self.clsConv.backpropagate(d_l, learn_rate, (proposal.getSetNum() * 2) + 1)

        print("Loss " + str(loss))

        return loss

    def mutate(self, mutate_rate):
        self.Conv3x3.mutate(mutate_rate)
        self.clsConv.mutate(mutate_rate)
        self.bboxConv.mutate(mutate_rate)
        pass







