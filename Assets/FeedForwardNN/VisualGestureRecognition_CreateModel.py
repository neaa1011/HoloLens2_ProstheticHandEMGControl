# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:41:07 2023

@author: Aaron
"""

import copy
import os
import glob
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import tf2onnx
import onnx

from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

class const():
    DATASET_PATHS = [
        "C:/Users/Aaron/Desktop/Uni/MCI/Masterarbeit/Python_VisualGestureRecognition/FeedForwardNN_V4/data/Fist_Gesture",
        "C:/Users/Aaron/Desktop/Uni/MCI/Masterarbeit/Python_VisualGestureRecognition/FeedForwardNN_V4/data/Pinch_Gesture",
        "C:/Users/Aaron/Desktop/Uni/MCI/Masterarbeit/Python_VisualGestureRecognition/FeedForwardNN_V4/data/Spread_Gesture",
        "C:/Users/Aaron/Desktop/Uni/MCI/Masterarbeit/Python_VisualGestureRecognition/FeedForwardNN_V4/data/Thumb_Up_Gesture"] #,
        #"C:/Users/Aaron/Desktop/Uni/MCI/Masterarbeit/Python_VisualGestureRecognition/FeedForwardNN_V4/data/Rest_Gesture"]
    
    MODEL_INFORMATION_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/data/InfoModel.txt"

    TMP_DATA_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/data/tmpdata.txt"
    EMG_TIME_LABEL_DATA_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/data/EMGTimeLabel.txt"

    MODEL_SAVE_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/model/GestureRecognition.hdf5"

    TFLITE_SAVE_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/model/GestureRecognition.tflite"
    ONNX_SAVE_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/model/GestureRecognition.onnx"
    PD_SAVE_PATH = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/model/GestureRecognition.pd"
    
    HAND_JOINTS = [
        'Wrist', 'ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal', 'ThumbTip',
        'IndexMetacarpal', 'IndexKnuckle', 'IndexMiddle', 'IndexDistal', 'IndexTip',
        'MiddleMetacarpal', 'MiddleKnuckle', 'MiddleMiddle', 'MiddleDistal', 'MiddleTip',
        'RingMetacarpal', 'RingKnuckle', 'RingMiddle', 'RingDistal', 'RingTip',
        'PinkyMetacarpal', 'PinkyKnuckle', 'PinkyMiddle', 'PinkyDistal', 'PinkyTip']

    NUM_CLASSES = 4
    AVG_NO = 15

def train_test_split(dataset, label_dataset, train_size=0.75):
    dataset = list(zip(dataset, label_dataset))
    np.random.shuffle(dataset)

    split_index = int(len(dataset)* train_size)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test

def ConvertGlobPosToRelPos(pos):
    tmp_pos_data = copy.deepcopy(pos)
    #print(tmp_pos_data)

    # Convert to relative coordinates (2D)
    base_x, base_y, base_z = 0, 0, 0
    for index in range(0, len(tmp_pos_data), 3):
        x, y, z = tmp_pos_data[index], tmp_pos_data[index + 1], tmp_pos_data[index + 2]
        if index == 0:
            base_x, base_y, base_z = x, y, z
            
        tmp_pos_data[index] = x - base_x
        tmp_pos_data[index + 1] = y - base_y
        tmp_pos_data[index + 2] = z - base_z

    
    # Normalize each component separately
    max_x = max(abs(tmp_pos_data[i]) for i in range(0, len(tmp_pos_data), 3))
    max_y = max(abs(tmp_pos_data[i + 1]) for i in range(0, len(tmp_pos_data), 3))
    max_z = max(abs(tmp_pos_data[i + 2]) for i in range(0, len(tmp_pos_data), 3))
    
    def normalize_(n):
        if abs(n) == max_x:
            return n / max_x
        elif abs(n) == max_y:
            return n / max_y
        else:
            return n / max_z
    
    tmp_pos_data = list(map(normalize_, tmp_pos_data))
    
    
    return tmp_pos_data

def ConvertGlobRotToRelRot(rot):
    tmp_rot_data = copy.deepcopy(rot)
    #print("********** Tmp Rot Data Raw **********")
    #print(tmp_rot_data)

    # Convert to relative coordinates (2D)
    base_x, base_y, base_z = 0, 0, 0
    for index in range(0, len(tmp_rot_data), 3):
        x, y, z = tmp_rot_data[index], tmp_rot_data[index + 1] , tmp_rot_data[index + 2]
        if index == 0:
            base_x, base_y, base_z = x, y, z

        tmp_rot_data[index] = x - base_x
        tmp_rot_data[index + 1] = y - base_y
        tmp_rot_data[index + 2] = z - base_z


    # Normalize each component separately
    max_x = max(abs(tmp_rot_data[i]) for i in range(0, len(tmp_rot_data), 3))
    max_y = max(abs(tmp_rot_data[i + 1]) for i in range(0, len(tmp_rot_data), 3))
    max_z = max(abs(tmp_rot_data[i + 2]) for i in range(0, len(tmp_rot_data), 3))
    
    def normalize_(n):
        if abs(n) == max_x:
            return n / max_x
        elif abs(n) == max_y:
            return n / max_y
        else:
            return n / max_z
    
    tmp_rot_data = list(map(normalize_, tmp_rot_data))
    
    return tmp_rot_data


def LoadData():
    dataset = []#np.empty((0, 150), dtype=np.float32)
    listdataset = []

    label_dataset = []
    label_mapping = {"Fist_Gesture": 0, "Pinch_Gesture": 1, "Spread_Gesture": 2, "Thumb_Up_Gesture": 3}
    
    for p, folder in enumerate(const.DATASET_PATHS):
        files = glob.glob(os.path.join(folder, "*.txt"))
        
        print(f"Load folder: {folder} with {len(files)} detected files")
        numSamples = 0
        
        pos_data = []
        rot_data = []
        
        for file in files:
            with open(file, 'r') as f:
                samples = json.load(f)
                
                # Calculate the number of samples in the file
                num_samples = len(samples)
                #print(f"Number of samples: {num_samples}")
                numSamples = numSamples + num_samples

                if num_samples == 0:
                    print(f"  No samples found in file: {file}")
                    continue
                
                # Iterate over each sample/"image"
                for i, sample in enumerate(samples):
                    joints = sample["Joints"]
                    time_stamp = sample["Time"]
                    gesture = sample["Gesture"]   
                    camera = sample["MainCamera"] 
                    
                    label = label_mapping[gesture]
                    position = []
                    rotation = []
                    
                    # Get Landmarks
                    for j, joint_name in enumerate(joints):
                        pos = joints[joint_name]["Position"]
                        rot = joints[joint_name]["Rotation"]
                       
                        position.append(pos[0])
                        position.append(pos[1])
                        position.append(pos[2])
                        
                        rotation.append(rot[0])
                        rotation.append(rot[1])
                        rotation.append(rot[2])

                    # Convert global rotations to relative
                    rotation = ConvertGlobRotToRelRot(rotation)
                    # Convert global coordinates to relative
                    position = ConvertGlobPosToRelPos(position)
                      
                    rot_data.append(rotation)                                                          
                    pos_data.append(position)   
                    
                    # Comment when used with AVG
                    #label_dataset.append(label)
                    
        print(f"Gesture {p} has {numSamples} Samples.")
        # Calculate Average
        noIterations = numSamples // const.AVG_NO
        print(f"New Sample Size: {noIterations}")
        avgPos_dataset = np.zeros((noIterations, len(pos_data[0])))
        avgRot_dataset = np.zeros((noIterations, len(rot_data[0])))
        
        for a in range(noIterations):
            start_index = a*const.AVG_NO
            end_index = (a+1)*const.AVG_NO
            
            avgPos_dataset[a] = np.mean(pos_data[start_index:end_index], axis=0)
            avgRot_dataset[a] = np.mean(rot_data[start_index:end_index], axis=0)
            
            # avgPos_dataset[a] = pos_data[start_index]
            # avgRot_dataset[a] = rot_data[start_index]
            
            label_dataset.append(label)
                
        
        #tmpdataset = np.hstack((pos_data, rot_data))
        tmpdataset = np.hstack((avgPos_dataset, avgRot_dataset))

        listdataset.append(tmpdataset)
        
           
    dataset = np.concatenate(listdataset, axis=0)
        
        
    dataset = np.array(dataset, dtype=np.float32)
    label_dataset = np.array(label_dataset, dtype=np.int32)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Label shape: {label_dataset.shape}")                  
    
    X_train, X_test, y_train, y_test = train_test_split(dataset, label_dataset, train_size=0.75)#, random_state=42)
    
    print(f"Total number of samples: {len(dataset)}, Number of training data: {len(X_train)}")
    return X_train, X_test, y_train, y_test  

def BuildModel():
    
    position_input = tf.keras.layers.Input((25 * 3,), name='position_input')
    rotation_input = tf.keras.layers.Input((25 * 3,), name='rotation_input')

    position_flatten = tf.keras.layers.Flatten()(position_input)
    rotation_flatten = tf.keras.layers.Flatten()(rotation_input)

    merged = tf.keras.layers.concatenate([position_flatten, rotation_flatten])
    dropout_1 = tf.keras.layers.Dropout(0.2)(merged)
    dense_2 = tf.keras.layers.Dense(75, activation='relu')(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.3)(dense_2)
    dense_3 = tf.keras.layers.Dense(20, activation='relu')(dropout_2)
    dropout_3 = tf.keras.layers.Dropout(0.4)(dense_3)
    dense_4 = tf.keras.layers.Dense(10, activation='relu')(dropout_3)
    output = tf.keras.layers.Dense(const.NUM_CLASSES, activation='softmax')(dense_4)

    model = tf.keras.models.Model(inputs=[position_input, rotation_input], outputs=output)

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

def TrainModel(model, X_train, y_train, X_test, y_test, cp_callback, es_callback):
    model.fit(    
        [X_train[:,:75], X_train[:,75:]],
        y_train,
        epochs=1000,
        batch_size=512,#128,
        validation_data=([X_test[:, :75], X_test[:, 75:]], y_test),
        callbacks=[cp_callback, es_callback]
    )

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    ax.set_title('Confusion Matrix Frame Data')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))


def accuracy_wrapper(estimator, X, y):
    y_pred = estimator.predict(X)

    # Predict using the model with both inputs
    y_pred = estimator.predict(X)
    
    return accuracy_score(y, np.argmax(y_pred, axis=1))

def f_importances(ANN_model, X_data, y_data, feature_names):
   # Calculate permutation importance
    perm_importance = permutation_importance(
        ANN_model, X_data, y_data, scoring=accuracy_wrapper, n_repeats=10, random_state=0
    )
    importances = perm_importance.importances_mean
    features = np.array(feature_names)
    # Sort feature importances
    sorted_idx = importances.argsort()
    feature_names_sorted = [feature_names[i] for i in sorted_idx]
    importances_sorted = importances[sorted_idx]
    
    # Print feature importances
    for feature, importance in zip(feature_names_sorted, importances_sorted):
        print(f"{feature}: {importance}")
    
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(features)))  # Increase figure height based on the number of features
    ax.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features ANN")
    ax.invert_yaxis()  # Invert the y-axis to display features from top to bottom
    plt.tight_layout()  # Adjust spacing between the bars and labels
    plt.show()
    plt.savefig('Feature_relevance_ann.png', dpi=300)
    
    # Print feature importances
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance}")
    
    # Plot feature importances
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.show()

def CrossValidateModel(X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    validation_accuracies = []
    best_model = None  # Initialize best_model to None

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}/{num_folds}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model, cp_cb, es_cb = BuildModel()  # Create or load your model here

        print("********** Train Model **********")
        TrainModel(model, X_train, y_train, X_val, y_val, cp_cb, es_cb)

        print("********** Evaluate Model **********")
        val_loss, val_acc = model.evaluate([X_val[:, :75], X_val[:, 75:]], y_val, batch_size=512)
        validation_accuracies.append(val_acc)
        
        # Save the model if it has the best accuracy so far (optional)
        if not validation_accuracies or val_acc > max(validation_accuracies):
            best_model = model


    # Calculate and print the average validation accuracy across all folds
    print(f"Average validation accuracy: {np.mean(validation_accuracies)}")
    
    # Check if any model was saved as the best model
    if best_model is None:
        print("Best model is none! Save last model")
        model.save(const.MODEL_SAVE_PATH)
    else:
        # Save the best-performing model
        best_model.save(const.MODEL_SAVE_PATH)


## MAIN
# Preprocess Dataset
print("********** Load Data **********")
X_train, X_test, y_train, y_test = LoadData()

# Cross-validation
print("********** Cross-validation **********")
#CrossValidateModel(X_train, y_train, num_folds=5)

# Build Model
print("********** Build Model **********")
#model, cp_cb, es_cb = BuildModel()
    
# Train Model
print("********** Train Model **********")
#TrainModel(model, X_train, y_train, X_test, y_test, cp_cb, es_cb)
    
# Loading the saved model
print("********** Load Model **********")
model = tf.keras.models.load_model(const.MODEL_SAVE_PATH)

# Model evaluation
print("********** Evaluate Model **********")
val_loss, val_acc = model.evaluate([X_test[:, :75], X_test[:, 75:]], y_test, batch_size=512)
print(f"Acuuracy: {val_acc} Loss: {val_loss}")

# Feature Importance
print("********** Feature Importance **********")
#print("** NEEDS TO BE INCLUDED **")
#feature_idx = list(range(0,75))
#f_importances(model, X_test, y_test, feature_idx)

# Write Info in Info File
with open(const.MODEL_INFORMATION_PATH, 'w') as file:
    file.write("{\"Accuracy\": \"" + str(val_acc) + "\", \"Loss\": \"" + str(val_loss)+"\"}")

# Inference test
predict_result = model.predict([np.expand_dims(X_test[0, :75], axis=0), np.expand_dims(X_test[0, 75:], axis=0)])
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


##Convert Model for Tensorflow-Lite
print("********** Convert Model To TFLite **********")
# Save as a model dedicated to inference
model.save(const.MODEL_SAVE_PATH, include_optimizer=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(const.TFLITE_SAVE_PATH, 'wb').write(tflite_quantized_model)

# Inference test
interpreter = tf.lite.Interpreter(model_path=const.TFLITE_SAVE_PATH)
interpreter.allocate_tensors()
# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Modify input shape to match the model's input shape
interpreter.resize_tensor_input(input_details[0]['index'], (1, 75))
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'],  np.array([X_test[0, :75]]))

# Inference implementation
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))


### Evaluation via Confusion Matrix
Y_pred = model.predict([X_test[:, :75], X_test[:, 75:]])
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)


##Convert Model for ONNX
print("********** Convert Model To onnx **********")
# Loading the saved model
model = tf.keras.models.load_model(const.MODEL_SAVE_PATH)
# Export the model
tf.saved_model.save(model, const.PD_SAVE_PATH)    
# convert model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=9)
onnx.save_model(onnx_model, const.ONNX_SAVE_PATH)


