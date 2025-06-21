import cv2
class FrameHandler:
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height
    def _grab_cam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame
    def handle_frame(self, frame):
        frame = self._grab_cam()

        return frame
