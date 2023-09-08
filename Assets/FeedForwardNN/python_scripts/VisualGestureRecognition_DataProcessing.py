# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:08:17 2023

@author: Aaron
"""

import os
import glob
import json
import numpy as np
import VisualGestureRecognition_Const as const
from sklearn.model_selection import train_test_split

def LoadData():
    pos_dataset = [] 
    label_dataset = []
    label_mapping = {"Fist_Gesture": 0, "Pinch_Gesture": 1, "Spread_Gesture": 2, "Thumb_Up_Gesture": 3}
    
    for i, folder in enumerate(const.DATASET_PATHS):
        files = glob.glob(os.path.join(folder, "*.txt"))
        print(f"Load folder: {folder} with {len(files)} detected files")
        for file in files:
            with open(file, 'r') as f:
                samples = json.load(f)
                
                # Calculate the number of samples in the file
                num_samples = len(samples)
                #print(f"Number of samples: {num_samples}")

                if num_samples == 0:
                    print(f"  No samples found in file: {file}")
                    continue
                
                # Iterate over each sample/"image"
                for i, sample in enumerate(samples):
                    joints = sample["Joints"]
                    time_stamp = sample["Time"]
                    gesture = sample["Gesture"]    
                    
                    label = label_mapping[gesture]
                    position = []
                    
                    # Get Landmarks
                    for j, joint_name in enumerate(joints):
                        pos = joints[joint_name]["Position"]
                        rot = joints[joint_name]["Rotation"]
                        
                        position.append(pos[0])
                        position.append(pos[1])
                        #position.append(pos[2])
                                          
                    pos_dataset.append(position)                    
                    label_dataset.append(label)
     
    pos_dataset = np.array(pos_dataset, dtype=np.float32)
    label_dataset = np.array(label_dataset, dtype=np.int32)
    print(f"Dataset shape: {pos_dataset.shape}")                
    
    X_train, X_test, y_train, y_test = train_test_split(pos_dataset, label_dataset, train_size=0.75, random_state=42)
    
    print(f"Total number of samples: {len(pos_dataset)}, Number of training data: {len(X_train)}")
    return X_train, X_test, y_train, y_test                       

                    
                    
                    
                
                                   
    
            
    
    

    
    
    