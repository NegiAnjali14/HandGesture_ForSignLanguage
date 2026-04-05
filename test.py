import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # hides TensorFlow logs
os.environ["GRPC_VERBOSITY"] = "ERROR"             # hides gRPC warnings

import warnings
warnings.filterwarnings("ignore")                  # hides Python warnings

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)     # hides absl/MediaPipe logs
#----------------------------------------------------------------------------------

import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

# ── Fix for DepthwiseConv2D 'groups' error ──────────────────────────────────
import tensorflow as tf
from tensorflow import keras

def load_model_safe(model_path, labels_path):
    """Load keras model with compatibility fix for Teachable Machine models."""
    # Patch: custom_object_scope fixes the 'groups' argument error
    with keras.utils.custom_object_scope({
        'DepthwiseConv2D': lambda **kwargs: keras.layers.DepthwiseConv2D(
            **{k: v for k, v in kwargs.items() if k != 'groups'}
        )
    }):
        model = keras.models.load_model(model_path, compile=False)

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return model, labels

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 300
OFFSET   = 20
LABELS   = ["Hello", "I Love You", "No", "Okay", "Please", "Thank You", "Yes"]

# ── Singletons ────────────────────────────────────────────────────────────────
_detector   = None
_model      = None
_labels     = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = HandDetector(maxHands=1)
    return _detector

def _get_model():
    global _model, _labels
    if _model is None:
        _model, _labels = load_model_safe(
            "model/keras_model.h5",
            "model/labels.txt"
        )
    return _model, _labels

# ── Core prediction function ──────────────────────────────────────────────────
def predict_gesture(frame: np.ndarray) -> dict:
    result = {
        "label":      "No Hand Detected",
        "confidence": 0.0,
        "hand_found": False,
        "annotated":  frame.copy(),
        "error":      None,
    }

    try:
        detector      = _get_detector()
        model, labels = _get_model()

        img_output = frame.copy()
        hands, _   = detector.findHands(frame.copy(), draw=False)

        if not hands:
            return result

        hand        = hands[0]
        x, y, w, h  = hand["bbox"]

        # Safe crop with boundary clamping
        y1 = max(0, y - OFFSET)
        y2 = min(frame.shape[0], y + h + OFFSET)
        x1 = max(0, x - OFFSET)
        x2 = min(frame.shape[1], x + w + OFFSET)

        img_crop = frame[y1:y2, x1:x2]
        if img_crop.size == 0:
            result["error"] = "Crop empty — move hand away from edges."
            return result

        # Build 300×300 white background
        img_white  = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        crop_h, crop_w = img_crop.shape[:2]
        aspect_ratio   = crop_h / max(crop_w, 1)

        if aspect_ratio > 1:
            k     = IMG_SIZE / crop_h
            w_cal = math.ceil(k * crop_w)
            img_r = cv2.resize(img_crop, (w_cal, IMG_SIZE))
            w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
            img_white[:, w_gap:w_cal + w_gap] = img_r
        else:
            k     = IMG_SIZE / max(crop_w, 1)
            h_cal = math.ceil(k * crop_h)
            img_r = cv2.resize(img_crop, (IMG_SIZE, h_cal))
            h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_r

        # ── Run model manually (no cvzone Classifier) ──────────────────────
        img_input  = cv2.resize(img_white, (224, 224))        # Teachable Machine uses 224×224
        img_input  = img_input.astype("float32") / 127.5 - 1  # Normalize to [-1, 1]
        img_input  = np.expand_dims(img_input, axis=0)        # Add batch dimension → (1,224,224,3)

        prediction = model.predict(img_input, verbose=0)[0]   # Shape: (num_classes,)
        index      = int(np.argmax(prediction))
        confidence = float(prediction[index])
        label      = labels[index] if index < len(labels) else LABELS[index]

        # ── Annotate frame ─────────────────────────────────────────────────
        cv2.rectangle(img_output,
                      (x - OFFSET, y - OFFSET - 60),
                      (x - OFFSET + 400, y - OFFSET + 10),
                      (0, 200, 100), cv2.FILLED)
        cv2.putText(img_output,
                    f"{label}  {confidence*100:.1f}%",
                    (x - OFFSET + 10, y - OFFSET - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)
        cv2.rectangle(img_output,
                      (x - OFFSET, y - OFFSET),
                      (x + w + OFFSET, y + h + OFFSET),
                      (0, 200, 100), 3)
        detector.findHands(img_output, draw=True)

        result.update({
            "label":      label,
            "confidence": confidence,
            "hand_found": True,
            "annotated":  img_output,
        })

    except Exception as exc:
        result["error"] = str(exc)

    return result

# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        res = predict_gesture(frame)
        if res["error"]:
            print(f"[WARN] {res['error']}")
        cv2.imshow("Hand Gesture — press q", res["annotated"])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    