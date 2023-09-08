
#if UNITY_EDITOR

using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Scripting.Python;
using UnityEngine;

[CustomEditor(typeof(TrainNN))]

public class TrainNN_Editor : Editor
{
    TrainNN targetTrainNN;

    private void OnEnable()
    {
        targetTrainNN = (TrainNN)target;
    }

    public override void OnInspectorGUI()
    {
        if (GUILayout.Button("Train Model", GUILayout.Height(35)))
        {
            CreateModel();
        }
        if (GUILayout.Button("Classify Current Tmp Data in Python", GUILayout.Height(35)))
        {
            ClassifyGesture();
        }
    }

    private void CreateModel()
    {
        Debug.Log("Create and Train Model");
        PythonRunner.RunFile($"{Application.dataPath}/FeedForwardNN/VisualGestureRecognition_CreateModel.py");
    }

    private void ClassifyGesture()
    {
        Debug.Log("Classify gesture on current tmp data");
        PythonRunner.RunFile($"{Application.dataPath}/FeedForwardNN/VisualGestureRecognition_MainLoop.py");
    }
}

#endif