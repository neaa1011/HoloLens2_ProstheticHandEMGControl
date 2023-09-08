# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:47:08 2023

@author: Aaron
"""

class const():
    DATASET_PATHS = [
        "./data/Fist_Gesture",
        "./data/Pinch_Gesture",
        "./data/Spread_Gesture",
        "./data/Thumb_Up_Gesture"]

    MODEL_SAVE_PATH = "model/GestureRecognition.hdf5"

    TFLITE_SAVE_PATH = "model/GestureRecognition.tflite"

    HAND_JOINTS = [
        'Wrist', 'ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal', 'ThumbTip',
        'IndexMetacarpal', 'IndexKnuckle', 'IndexMiddle', 'IndexDistal', 'IndexTip',
        'MiddleMetacarpal', 'MiddleKnuckle', 'MiddleMiddle', 'MiddleDistal', 'MiddleTip',
        'RingMetacarpal', 'RingKnuckle', 'RingMiddle', 'RingDistal', 'RingTip',
        'PinkyMetacarpal', 'PinkyKnuckle', 'PinkyMiddle', 'PinkyDistal', 'PinkyTip']

    NUM_CLASSES = 4
