from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval

gt_file = './annotations/crowdpose_val.json'
preds = './annotations/preds.json'

cocoGt = COCO(gt_file)
cocoDt = cocoGt.loadRes(preds)
cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
