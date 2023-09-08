using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ClassifyGesture
{
    public PythonManager pyManager;

    // Update is called once per frame
    public void ClassifyCurrentGesture()
    {
        Debug.Log("*** Classification: Call Python Manager");
        pyManager = new PythonManager();
        pyManager.ClassifyGesture();
    }
}
