"""
datacollection.py — Gesture Image Capture Tool
Hand Gesture Recognition System for Sign Language
Author: Anjali Negi & Divyansh Agrawal | BCA Final Year Project

Usage:
    python datacollection.py --gesture Hello
    Press 's' to save a frame, 'q' to quit.
"""

import cv2
import numpy as np
import math
import time
import os
import argparse
from cvzone.HandTrackingModule import HandDetector

# ─── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE = 300
OFFSET   = 20
GESTURES = ["Hello", "I Love You", "No", "Okay", "Please", "Thank You", "Yes"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand gesture images for training.")
    parser.add_argument(
        "--gesture", type=str, default="Hello",
        choices=GESTURES,
        help="Name of the gesture to collect (default: Hello)"
    )
    parser.add_argument(
        "--target", type=int, default=300,
        help="Number of images to collect (default: 300)"
    )
    return parser.parse_args()


def main():
    args    = parse_args()
    folder  = f"Data/{args.gesture}"
    os.makedirs(folder, exist_ok=True)

    cap      = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    counter  = 0

    print(f"\n[INFO] Collecting images for gesture: '{args.gesture}'")
    print(f"[INFO] Target: {args.target} images  →  Saving to: {folder}/")
    print("[INFO] Press 's' to SAVE a frame  |  'q' to QUIT\n")

    while True:
        success, img = cap.read()
        if not success:
            continue

        hands, img = detector.findHands(img)
        img_white  = None

        if hands:
            hand        = hands[0]
            x, y, w, h = hand["bbox"]

            y1 = max(0, y - OFFSET)
            y2 = min(img.shape[0], y + h + OFFSET)
            x1 = max(0, x - OFFSET)
            x2 = min(img.shape[1], x + w + OFFSET)

            img_crop = img[y1:y2, x1:x2]
            if img_crop.size == 0:
                continue

            img_white   = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            crop_h, crop_w = img_crop.shape[:2]
            aspect_ratio   = crop_h / max(crop_w, 1)

            if aspect_ratio > 1:
                k     = IMG_SIZE / crop_h
                w_cal = math.ceil(k * crop_w)
                img_r = cv2.resize(img_crop, (w_cal, IMG_SIZE))
                w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
                img_white[:, w_gap : w_cal + w_gap] = img_r
            else:
                k     = IMG_SIZE / max(crop_w, 1)
                h_cal = math.ceil(k * crop_h)
                img_r = cv2.resize(img_crop, (IMG_SIZE, h_cal))
                h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
                img_white[h_gap : h_cal + h_gap, :] = img_r

            cv2.imshow("Cropped", img_crop)
            cv2.imshow("White Background", img_white)

        # HUD overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, f"Gesture: {args.gesture}   Saved: {counter}/{args.target}",
                    (10, 33), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 180), 2)

        cv2.imshow("Data Collection  |  s = save   q = quit", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and img_white is not None:
            filename = f"{folder}/img_{time.time():.0f}.jpg"
            cv2.imwrite(filename, img_white)
            counter += 1
            print(f"  Saved [{counter:>4}] → {filename}")
            if counter >= args.target:
                print(f"\n[DONE] Collected {counter} images for '{args.gesture}'.")
                break
        elif key == ord("q"):
            print("\n[QUIT] Data collection stopped early.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
