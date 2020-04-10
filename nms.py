from utils import bb_intersection_over_union
import numpy as np

def non_max_suppression_slow(proposals, overlapThresh):

	if len(proposals) == 0:
		return []

	proposal_list_and_cls = []
	boxes = []
	for i in proposals:
		proposal_list_and_cls.append((i, i.getCls()))

	proposal_list_and_cls = sorted(proposal_list_and_cls, key=lambda tup: tup[1])

	for i in range(1, 51):
		if proposal_list_and_cls[-i][1] > 0.99:
			boxes.append((proposal_list_and_cls[-i][0].getPoints(), proposal_list_and_cls[-i][1]))

	picked_proposals = []

	while len(boxes) > 0:
		indexes_to_delete = []
		best = boxes[0][0]
		picked_proposals.append(best)
		boxes.pop(0)
		for index, box in enumerate(boxes):
			if bb_intersection_over_union(best, box[0]) > overlapThresh:
				indexes_to_delete.append(index)

		boxes = [i for j, i in enumerate(boxes) if j not in indexes_to_delete]

	return picked_proposals
