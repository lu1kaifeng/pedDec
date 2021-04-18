import cv2
import numpy as np
import threading
import base64
from typing import List

from deep_sort.sort.track import Track


class Inference:
    def __init__(self, config, threshold=0.2):
        self.transforms = config['transforms']
        self.model = config['model']
        self.data_source = config['data_source']
        self.tracker = config['tracker']
        self.threshold = threshold
    @staticmethod
    def _track_to_dict(tracks:List[Track] ):
        l = []
        for t in tracks:
            l.append({
                'id':t.track_id,
                'image':bytes(cv2.imencode('.jpg',t.img)[1]),
                'age':t.age,
                'confidence':t.confidence
            })
        return l

    def run_inference(self,on_frame_callback,on_track_callback):
        def worker():
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
                on_frame_callback(bytes(cv2.imencode('.jpg',im)[1]))
                on_track_callback(self._track_to_dict(self.tracker.tracks()))
        threading.Thread(target=worker).start()

    def pedestrian_record(self):
        return self.tracker.tracks()
