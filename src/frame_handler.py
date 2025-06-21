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

    # --------------------------- public helpers --------------------------- #
    def extract_face(self, frame: np.ndarray) -> np.ndarray:
        """Return the input frame with only the face region preserved.

        Args:
            frame: BGR image (OpenCV default).
        Returns:
            BGR image where pixels outside the face are set to zero.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame  # Gracefully fallback when no face found

        h, w = frame.shape[:2]
        # Collect all 468 landmark points
        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in results.multi_face_landmarks[0].landmark],
            dtype=np.int32,
        )

        # Build convex hull around the landmarks and fill it to create a mask
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Feather edges for smoother compositing
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.person_seg.process(rgb)
        mask_prob = results.segmentation_mask
        if mask_prob is None:
            return frame  # Fallback

        mask = (mask_prob > threshold).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

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
        # Extract face first to ensure it's always preserved
        face_extracted = self.extract_face(frame)
        
        # Obtain person mask (same logic as in `extract_person`).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.person_seg.process(rgb)
        mask_prob = results.segmentation_mask
        if mask_prob is None:
            # If model fails, just return face extracted frame.
            return face_extracted

        # Binary mask where 255 denotes person pixels.
        mask = (mask_prob > 0.25).astype(np.uint8) * 255
        # Smooth edges for nicer compositing.
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # Prepare 3-channel versions of mask and its inverse.
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        inv_mask_3 = cv2.bitwise_not(mask_3)

        # Background canvas: bright green.
        green_bg = np.full_like(frame, (0, 255, 0))

        # Keep person area from original frame.
        person_part = cv2.bitwise_and(frame, mask_3)
        # Keep background area from green canvas.
        bg_part = cv2.bitwise_and(green_bg, inv_mask_3)

        # Combine person and background.
        combined = cv2.add(person_part, bg_part)
        
        # Overlay face extracted area to ensure face is always preserved
        # Convert face mask to 3-channel and use it to blend
        face_mask = cv2.cvtColor(cv2.cvtColor(face_extracted, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        face_mask = (face_mask > 0).astype(np.uint8) * 255
        
        # Blend face area with combined result
        face_area = cv2.bitwise_and(face_extracted, face_mask)
        non_face_area = cv2.bitwise_and(combined, cv2.bitwise_not(face_mask))
        
        return cv2.add(face_area, non_face_area)

    # --------------------------- compatibility --------------------------- #
    def handle_frame(self, frame: np.ndarray) -> np.ndarray:
        """Default behaviour kept for backward compatibility (face extraction)."""
        return self.make_background(frame)
