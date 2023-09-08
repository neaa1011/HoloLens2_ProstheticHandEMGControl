# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:41:46 2023

@author: Aaron
"""
import tensorflow as tf
import VisualGestureRecognition_Const as const

def BuildModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((25 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(const.NUM_CLASSES, activation='softmax')
    ])
    
    model.summary()
    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    const.MODEL_SAVE_PATH, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    
    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, cp_callback, es_callback
