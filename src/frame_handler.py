import cv2
import numpy as np
import mediapipe as mp


class FrameHandler:
    """Utility class that can extract either the face region or the full person mask
    from a video frame using MediaPipe models.
    """

    def __init__(self) -> None:
        # Face-mesh model for fine-grained face landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Selfie-segmentation model for full-body mask
        self.mp_selfie_seg = mp.solutions.selfie_segmentation
        # model_selection=1 → general model (full body incl. hair)
        self.person_seg = self.mp_selfie_seg.SelfieSegmentation(model_selection=1)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _get_face_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return binary (0/255) mask of the face region."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return np.zeros((h, w), dtype=np.uint8)

        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in results.multi_face_landmarks[0].landmark],
            dtype=np.int32,
        )
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Slight smoothing and re-thresholding to remove jagged edges while keeping binary mask.
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _get_person_mask(self, frame: np.ndarray, threshold: float = 0.25) -> np.ndarray:
        """Return binary (0/255) mask of the person region (full body)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.person_seg.process(rgb)
        if results.segmentation_mask is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        mask_prob = results.segmentation_mask
        mask = (mask_prob > threshold).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    # --------------------------- public helpers --------------------------- #
    def extract_face(self, frame: np.ndarray) -> np.ndarray:
        """Return the input frame with only the face region preserved.

        Args:
            frame: BGR image (OpenCV default).
        Returns:
            BGR image where pixels outside the face are set to zero.
        """
        mask = self._get_face_mask(frame)
        if mask.sum() == 0:
            return frame
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(frame, mask_3)

    def extract_person(self, frame: np.ndarray, threshold: float = 0.25) -> np.ndarray:
        """Return the input frame with only the person (head + torso + hair) preserved.

        Args:
            frame: BGR image (OpenCV default).
            threshold: Probability cut-off for the segmentation mask.
        Returns:
            BGR image where pixels outside the person are set to zero.
        """
        mask = self._get_person_mask(frame, threshold)
        if mask.sum() == 0:
            return frame
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(frame, mask_3)

    def make_background(self, frame: np.ndarray) -> np.ndarray:
        """Replace everything *except* the person with solid green background.

        The method uses the same selfie-segmentation model as `extract_person` but
        composites the original person pixels over a pure green canvas
        (0, 255, 0 in BGR). Face area is always preserved using face extraction.

        Args:
            frame: BGR image from camera.

        Returns:
            BGR image where the person remains and the background is green.
        """
        # Calculate masks
        face_mask = self._get_face_mask(frame)
        person_mask = self._get_person_mask(frame)

        # If person mask fails, fallback to face mask only.
        if person_mask.sum() == 0:
            person_mask = face_mask.copy()

        # Combine masks to ensure both face and body are kept.
        combined_mask = cv2.bitwise_or(face_mask, person_mask)

        mask_3 = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        inv_mask_3 = cv2.bitwise_not(mask_3)

        # Compose foreground and green background.
        green_bg = np.full_like(frame, (0, 255, 0))
        fg = cv2.bitwise_and(frame, mask_3)
        bg = cv2.bitwise_and(green_bg, inv_mask_3)
        return cv2.add(fg, bg)

    # --------------------------- compatibility --------------------------- #
    def handle_frame(self, frame: np.ndarray) -> np.ndarray:
        """Default behaviour kept for backward compatibility (face extraction)."""
        return self.make_background(frame)
