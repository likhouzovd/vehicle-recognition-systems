from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo


class ObjectDetector():
    def __init__(self):
        threshold = 0.9
        model_path = "detection_model.pth"

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = 600
        cfg.INPUT.MAX_SIZE_TEST = 800
        cfg.INPUT.FORMAT = 'BGR'
        cfg.TEST.DETECTIONS_PER_IMAGE = 20

        self.predictor = DefaultPredictor(cfg)

    def apply(self, im):
        outputs = self.predictor(im)
        ans = []
        for x in outputs["instances"]._fields["pred_boxes"]:
            x, y, w, h = int(x[0]), int(x[1]), int(x[2]) - int(x[0]), int(x[3]) - int(x[1])
            # if w * h > 0:
            ans.append([x, y, w, h])
        return ans
