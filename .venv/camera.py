import cv2
import torch
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes = [0, 1, 2, 3, 5, 6, 7, 9, 11, 15, 16]
        self.allowed_class_ids = [0, 1, 2, 3, 5, 6, 7, 9, 11, 15, 16]
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            results = self.model(frame)
            for result in results.xyxy[0]:
                class_id = int(result[5].item())
                if class_id in self.allowed_class_ids: #if result[5].item()
                    confidence = float(result[4].item())
                    label = self.model.names[class_id]
                    frame = cv2.rectangle(frame, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (int(result[0]), int(result[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


"""import cv2

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()"""