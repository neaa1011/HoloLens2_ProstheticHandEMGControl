using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.Scripting;
using Python.Runtime;
using UnityEditor.Scripting.Python;

public class PythonManager
{
    [MenuItem("PythonManager/Run")]

    public static void Run()
    {
        PythonRunner.EnsureInitialized();
        using (Py.GIL())
        {
            try
            {
                dynamic sys = Py.Import("sys");
                UnityEngine.Debug.Log($"python version: {sys.version}");
            }
            catch (PythonException e)
            {
                UnityEngine.Debug.LogException(e);
            }
        }
    }

    public void ClassifyGesture()
    {
        Debug.Log("*** Classification: Call Python script");
        PythonRunner.RunFile($"{Application.dataPath}/FeedForwardNN/VisualGestureRecognition_MainLoop.py");
    }
}
