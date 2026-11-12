"""Shared landmark normalization and augmentation used by training and live inference."""

import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3
NO_HAND_LABEL = "nothing"


def normalize_landmarks(landmarks):
    """Translate to the wrist, scale by hand size, and flatten to 63 features."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    points = points - points[0]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return (points / scale).reshape(-1).astype(np.float32)


def rotate_xy(points, degrees):
    radians = np.deg2rad(degrees)
    cos_a, sin_a = np.cos(radians), np.sin(radians)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T
    return points


def augment_landmarks(features, rng):
    """Random rotation, scale jitter, and coordinate noise for training only."""
    points = features.reshape(NUM_LANDMARKS, 3).copy()
    points = rotate_xy(points, float(rng.uniform(-12.0, 12.0)))
    points *= float(rng.uniform(0.9, 1.1))
    points[:, :2] += rng.normal(0.0, 0.01, size=(NUM_LANDMARKS, 2))
    points[:, 2] += rng.normal(0.0, 0.01, size=NUM_LANDMARKS)
    return points.reshape(-1).astype(np.float32)


def landmarks_from_mediapipe(hand_landmarks, frame_width, frame_height):
    """Convert MediaPipe normalized landmarks to the aspect-correct training convention."""
    max_side = float(max(frame_width, frame_height))
    points = np.array(
        [
            [point.x * frame_width, point.y * frame_height, point.z * frame_width]
            for point in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )
    return points / max_side
