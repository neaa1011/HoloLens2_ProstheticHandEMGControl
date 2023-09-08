# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:41:07 2023

@author: Aaron
"""
import copy
import json
import time
import numpy as np
import tensorflow as tf

# ONLY WHEN RUNNING IN UNITY
import UnityEngine as ue
import TMPro as tp

class const():
    DATASET_PATHS = [
        "C:/Users/Aaron/AppData/LocalLow/DefaultCompany/MyFirstProject/Fist_Gesture",
        "C:/Users/Aaron/AppData/LocalLow/DefaultCompany/MyFirstProject/Pinch_Gesture",
        "C:/Users/Aaron/AppData/LocalLow/DefaultCompany/MyFirstProject/Spread_Gesture",
        "C:/Users/Aaron/AppData/LocalLow/DefaultCompany/MyFirstProject/Thumb_Up_Gesture"]
    
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

    NUM_CLASSES = 5



class GestureClassifier(object):
    def __init__(
        self,
        model_path=const.TFLITE_SAVE_PATH,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, data):      
        position_list = data.tolist()[0][:75]
        rotation_list = data.tolist()[0][75:]

    
        position_input_index = self.input_details[0]['index']
        rotation_input_index = self.input_details[1]['index']
        input_shape = self.input_details[0]['shape']
    
        if len(position_list) != input_shape[1] or len(rotation_list) != input_shape[1]:
            raise ValueError(f"Invalid input size. Expected {input_shape[1]} elements for both position and rotation, "
                             f"but got {len(position_list)} and {len(rotation_list)} respectively.")
    
        position_input_data = np.array(position_list, dtype=np.float32)
        position_input_data = np.reshape(position_input_data, (1, 75))
        rotation_input_data = np.array(rotation_list, dtype=np.float32)
        rotation_input_data = np.reshape(rotation_input_data, (1, 75))
    
        self.interpreter.set_tensor(position_input_index, position_input_data)
        self.interpreter.set_tensor(rotation_input_index, rotation_input_data)
        self.interpreter.invoke()
    
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
    
        result_index = np.argmax(np.squeeze(result))
        probability = np.max(np.squeeze(result))
    
        return result_index, probability
    
    
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

    #print(tmp_pos_data)

    # Normalization
    max_value = max(list(map(abs, tmp_pos_data)))
    def normalize_(n):
        return n / max_value
    
    tmp_pos_data = list(map(normalize_, tmp_pos_data))
    #print(tmp_pos_data)
    
    return tmp_pos_data

def ConvertGlobRotToRelRot(rot):
    tmp_rot_data = copy.deepcopy(rot)
    #print("********** Tmp Rot Data Raw **********")
    #print(tmp_rot_data)

    # Convert to relative coordinates (2D)
    base_x, base_y, base_z = 0, 0, 0
    for index in range(0, len(tmp_rot_data), 3):
        x, y, z = tmp_rot_data[index], tmp_rot_data[index + 1], tmp_rot_data[index + 2]
        if index == 0:
            base_x, base_y, base_z = x, y, z

        tmp_rot_data[index] = x - base_x
        tmp_rot_data[index + 1] = y - base_y
        tmp_rot_data[index + 2] = z - base_z

    #print("********** Tmp Rot Data Relative **********")
    #print(tmp_rot_data)

    # Normalization
    max_value = max(list(map(abs, tmp_rot_data)))
    def normalize_(n):
        return n / max_value
    
    tmp_rot_data = list(map(normalize_, tmp_rot_data))
    #print("********** Tmp Rot Data Normalized**********")
    #print(tmp_rot_data)
    
    return tmp_rot_data

def LoadTmpData():
    tmp_data = []
    tmp_pos_data = [] 
    tmp_rot_data = [] 
    pos = []
    rot = []
    
    print(f"Load folder: {const.TMP_DATA_PATH}")
    with open(const.TMP_DATA_PATH, 'r') as file:
        tmpdata = json.load(file)
        #print(f"Data: {tmpdata}") 

        if len(tmpdata) == 0:
            print(f"No samples found in tmp data file")
            

    joints = tmpdata["Joints"]
    time_stamp = tmpdata["time"]  
                    
    # Get Landmarks
    for j, joint_name in enumerate(joints):
                pos = joints[joint_name]["Position"]
                rot = joints[joint_name]["Rotation"]
                            
                tmp_pos_data.append(pos[0])
                tmp_pos_data.append(pos[1])
                tmp_pos_data.append(pos[2])
                
                tmp_rot_data.append(rot[0])
                tmp_rot_data.append(rot[1])
                tmp_rot_data.append(rot[2])

    # Convert global rotations to relative
    #tmp_rot_data = ConvertGlobRotToRelRot(tmp_rot_data)
    # Convert global coordinates to relative
    #tmp_pos_data = ConvertGlobPosToRelPos(tmp_pos_data)
            
    tmp_data.append(tmp_rot_data + tmp_pos_data)

    #dataset = np.array(tmp_pos_data, dtype=np.float32)
    #dataset = np.array(tmp_rot_data, dtype=np.float32)
    dataset = np.array(tmp_data, dtype=np.float32)
    
    print(f"Dataset shape: {dataset.shape}")
    #print(f"Data: {tmp_pos_data}")                
    
    return  dataset

# Load model
mGestureClassifier = GestureClassifier()

print("********** Load Data **********")
tmp = LoadTmpData()

print("********** Classify Gesture **********")
GestureID, GestureProbability = mGestureClassifier(tmp)
#Increase Gesture because currently None = ID 0 doesnt exist
GestureID = GestureID + 1 
print(f"Gesture ID is: {GestureID} with the probability of: {GestureProbability}")

# Write Gesture ID, time and probability in Info File
currentTime = time.strftime("%H:%M:%S", time.localtime())
print(currentTime)

new_data = {
    "Time": currentTime,
    "GestureID": str(GestureID),
    "Probability": str(GestureProbability)
}
print(new_data)

# Load existing JSON data from the file
existing_data = []
try:
    with open(const.EMG_TIME_LABEL_DATA_PATH, 'r') as file:
        existing_data = json.load(file)
except FileNotFoundError:
    pass

# Append the new data to the existing data
existing_data.append(new_data)

# Write the updated JSON data back to the file
with open(const.EMG_TIME_LABEL_DATA_PATH, 'w') as file:
    json.dump(existing_data, file)
    # file.write("{\"Time\": \"" + currentTime + "\", \"GestureID\": \"" + str(GestureID)+ "\", \"Probability\": \"" + str(GestureProbability)+"\"}\n")


label_mapping = {"Rest_Gesture": 0,"Fist_Gesture": 1, "Pinch_Gesture": 2, "Spread_Gesture": 3, "Thumb_Up_Gesture": 4}

# ONLY WHEN RUNNING IN UNITY
ue.Debug.Log("Python: Classified Gesture is: "+ str(GestureID)+" with Probability of: " +str(GestureProbability))
print("********** Find Text DetectedGestureID **********")
gOb = ue.GameObject.Find("DetectedGestureID")
if gOb is None:
    ue.Debug.Log("Python: Gameobject was not found")
    
print("********** Get Text DetectedGestureID **********")
myText = gOb.GetComponent[tp.TextMeshPro]()
if myText is None:
    ue.Debug.Log("Python: Text was not found")

if GestureProbability > 0.7:
    if GestureID == 0:
        ue.Debug.Log("********** None_Gesture **********")
        print("********** Set Text DetectedGestureID **********")
        myText.text = "None"
    if GestureID == 1:
        ue.Debug.Log("********** Fist_Gesture **********")
        print("********** Set Text DetectedGestureID **********")
        myText.text = "Fist"
    if GestureID == 2:
        ue.Debug.Log("********** Pinch_Gesture **********")
        myText.text = "Pinch"
    if GestureID == 3:
        ue.Debug.Log("********** Spread_Gesture **********")
        myText.text = "Spread"
    if GestureID == 4:
        ue.Debug.Log("********** Thumb_Up_Gesture **********")
        myText.text = "Thumb up"
else:
    myText.text = "None"