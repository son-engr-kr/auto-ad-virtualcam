import cv2
import numpy as np
import mediapipe as mp

class FrameHandler:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # static_image_mode=False → video; refine_landmarks=True → iris/lips refinement
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def handle_frame(self, frame):
        # Convert BGR(OpenCV default) to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame  # no face detected → return original

        h, w = frame.shape[:2]
        # Get pixel coordinates of the 468 landmarks
        pts = np.array([[int(lm.x * w), int(lm.y * h)]
                        for lm in results.multi_face_landmarks[0].landmark],
                       dtype=np.int32)

        # Build a convex hull around all landmark points
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Optional: feather edges for smoother blending
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        segmented = cv2.bitwise_and(frame, mask_3)

        return segmented
