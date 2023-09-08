using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MySceneManagement : MonoBehaviour
{
    public void StartRecordingGestures()
    {
        Debug.Log("********** Go to Recording **********");
        SceneManager.LoadScene("RecordScene");
    }

    public void StartEMGControl()
    {
        Debug.Log("********** Go to EMG Control **********");
        SceneManager.LoadScene("EMGControl");
    }

    public void StartClassifyingGestures()
    {
        Debug.Log("********** Go to Classify Gesture **********");
        SceneManager.LoadScene("ClassifyScene");
    }

    public void ExitScene()
    {
        Debug.Log("********** Go to Main Menu **********");
        SceneManager.LoadScene("MainMenu");
    }
}
