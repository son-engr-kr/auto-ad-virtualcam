import colorsys
import numpy as np
import pyvirtualcam
import cv2
from frame_handler import FrameHandler
if __name__ == "__main__":
    # Get camera properties
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")


    frame_handler = FrameHandler()

    with pyvirtualcam.Camera(width=width, height=height, fps=20) as cam:
        print(f'Using virtual camera: {cam.device}')
        frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
        while True:
            ret, frame = cap.read()
            # Convert BGR to RGB (reverse the color channels)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            frame = frame_handler.handle_frame(frame)

            # h, s, v = (cam.frames_sent % 100) / 100, 1.0, 1.0
            # r, g, b = colorsys.hsv_to_rgb(h, s, v)
            # # Mix the color with the original frame instead of overwriting
            # frame = cv2.addWeighted(frame, 0.7, np.full_like(frame, (r * 255, g * 255, b * 255)), 0.3, 0)

            
            cam.send(frame)
            cam.sleep_until_next_frame()
    cap.release()