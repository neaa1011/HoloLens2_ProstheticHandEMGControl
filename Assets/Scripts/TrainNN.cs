using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;

public class TrainNN : MonoBehaviour
{
    public class ModelInformation
    {
        public string Accuracy;
        public string Loss;
    }

    private GameObject txt;
    private string path = "./Assets/FeedForwardNN/data/InfoModel.txt";

    void Start()
    {
        txt = GameObject.Find("MyTxt");
        if(txt != null)
            LoadInfoOfNeuralNetwork();
    }
    public void LoadInfoOfNeuralNetwork()
    {
        Debug.Log("********** Load Model Information **********");
        string data = File.ReadAllText(path);
        ModelInformation modelInfo = JsonUtility.FromJson<ModelInformation>(data);
        Debug.Log("Accuracy: " + modelInfo.Accuracy);
        Debug.Log("Loss: " + modelInfo.Loss);

        float tmp = float.Parse(modelInfo.Accuracy);
        modelInfo.Accuracy = string.Format("{0:F3}", tmp);

        txt.SetActive(true);
        TextMeshPro newTxt = txt.transform.Find("AccTxt").GetComponent<TextMeshPro>();
        newTxt.text = modelInfo.Accuracy;
    }
}
