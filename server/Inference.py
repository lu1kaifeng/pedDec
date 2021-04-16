import cv2
import numpy as np


class Inference:
    def __init__(self, config, threshold=0.2):
        self.transforms = config['transforms']
        self.model = config['model']
        self.data_source = config['data_source']
        self.tracker = config['tracker']
        self.threshold = threshold

    def run_inference(self, callback):
        for i in self.data_source:
            image_name = i[0]
            im = cv2.imread(image_name)
            result = self.model.predict(im)
            font = cv2.FONT_HERSHEY_SIMPLEX
            result = list(filter(lambda x: x['score'] > self.threshold, result))
            bboxes = np.array(list(map(lambda v: np.array(v['bbox']), result)))
            confidence = list(map(lambda v: v['score'], result))
            track = self.tracker.update(bboxes, confidence, im)
            for value in track:
                x, y, w, h, track, conf = value
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(im, '行人{:d}'.format(track),
                            (x, y), font, 0.5, (255, 0, 0), thickness=2)
            callback(im)

    def pedestrian_record(self):
        return self.tracker.tracks()
