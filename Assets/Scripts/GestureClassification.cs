using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.OpenXR;
using TMPro;
using Unity.Barracuda;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using System;
using Newtonsoft.Json.Linq;


#if !UNITY_EDITOR && UNITY_METRO
using System.Threading.Tasks;
using Windows.Storage;
#endif

public class GestureClassification : MonoBehaviour
{


    private readonly HandJointLocation[] handJointLocations = new HandJointLocation[HandTracker.JointCount];

    private TrackedHandside trackedHand;
    public HandSideManager myCurrHandSide;
    public TextMeshPro detectedGestureID;
    public NNModel modelAsset;

    /* Start TCP Socket */
    public TCPConnectionToESP SocketClient;
    private bool sentFlag;
    /* End TCP Socket */

#if UNITY_EDITOR
    /* Start PythonNet*/
    private ClassifyGesture myGestureClassification;
    //TMP_DATA_PATH = "./Assets/FeedForwardNN/data/tmpdata.txt";
    /* End PythonNet */
#elif !UNITY_EDITOR && UNITY_METRO
    /* Start Barracuda */
    public struct Prediction
    {
        // The most likely value for this prediction
        public int predictedValue;
        public float probability;
        // The list of likelihoods for all the possible classes
        public float[] predicted;

        public void SetPrediction(Tensor t)
        {
            // Extract the float value outputs into the predicted array.
            predicted = t.AsFloats();
            // The most likely one is the predicted value.
            predictedValue = Array.IndexOf(predicted, predicted.Max());
            probability = predicted.Max();
            //Increase Value, None gesture not implemented
            predictedValue++;
            Debug.Log($"Predicted {predictedValue}, Probability {probability}");
        }
    }

    public struct GestureData
    {
        public string time;
        public Dictionary<string, JointData> Joints;
    }

    public class JointData
    {
        public List<float> Position;
        public List<float> Rotation;
    }

    public Prediction prediction;
    
    private Model m_RuntimeModel;
    private IWorker _engine;
    private string TMP_DATA_PATH;// = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/data/tmpdata.txt";
    private string tmp_data_file = "FNN/tmpdata.txt";
    private string emg_label_data_file = "FNN/EMGTimeLabel.txt";
    private string EMG_TIME_LABEL_DATA_PATH = "FNN";// = "C:/Users/Aaron/AR_Samples/V2_Masterthesis_GestureRecognition/Assets/FeedForwardNN/data/EMGTimeLabel.txt";
    /* End Barracuda */
#endif

    private void Start()
    {
#if UNITY_EDITOR
        /* Start PythonNet */
        myGestureClassification = new ClassifyGesture();
        /* End PythonNet */
#elif !UNITY_EDITOR && UNITY_METRO
        /* Start Barracuda */
        Debug.Log("********** Create Directory **********");
        EMG_TIME_LABEL_DATA_PATH = Path.Combine(ApplicationData.Current.RoamingFolder.Path, EMG_TIME_LABEL_DATA_PATH); /*****  "./Assets/FeedForwardNN/data";*/
        if (!Directory.Exists(EMG_TIME_LABEL_DATA_PATH))
        {
            Directory.CreateDirectory(EMG_TIME_LABEL_DATA_PATH);
        }
        TMP_DATA_PATH = Path.Combine(ApplicationData.Current.RoamingFolder.Path, tmp_data_file); /***** "./Assets/FeedForwardNN/data/tmpdata.txt"; */
        EMG_TIME_LABEL_DATA_PATH =  Path.Combine(ApplicationData.Current.RoamingFolder.Path, emg_label_data_file); /***** "./Assets/FeedForwardNN/data/EMGTimeLabel.txt";*/

        Debug.Log("********** Load Model **********");
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        _engine = WorkerFactory.CreateWorker(m_RuntimeModel, WorkerFactory.Device.GPU);
        prediction = new Prediction();
        /* End Barracuda */

        #endif
        trackedHand = myCurrHandSide.GetCurrentHandSide();

        /* Start TCP Connection */
        sentFlag = false;
        /* End TCP Connection */

        Debug.Log("********** Start Classification **********");
        InvokeRepeating("StartGestureClassification", 2f, 0.25f);
    }

    private void OnDestroy()
    {
#if !UNITY_EDITOR && UNITY_METRO
        _engine?.Dispose();
#endif
    }

    private void ClassifyGesturesRoutine()
    {
        Debug.Log("********** Classification Routine **********");
#if UNITY_EDITOR
        /* Start PythonNet */
        myGestureClassification.ClassifyCurrentGesture();
        /* End PythonNet */
#elif !UNITY_EDITOR && UNITY_METRO
        /* Start Barracuda */
        Debug.Log("********** Load Data **********");
        List<float> tmpData = LoadTmpData();

        Debug.Log("********** Classify Gesture **********");
        //Debug.Log("Create Tensor Input");
        //var inputX = new Tensor(1, 1, 1, 150, tmpData.ToArray());
        //Debug.Log("Create Tensor Output");
        //Tensor outputY = _engine.Execute(inputX).PeekOutput();

        Debug.Log("********** Split Data in Rotation and Position **********");
        var inputData = tmpData.ToArray();
        float[] positionData = inputData.Take(75).ToArray();
        float[] rotationData = inputData.Skip(75).ToArray();

        // Create position and rotation tensors
        Debug.Log("********** Create Input Tensor **********");
        var positionTensor = new Tensor(1, 1, 1, 75, positionData);
        var rotationTensor = new Tensor(1, 1, 1, 75, rotationData);

        // Create dictionary for input tensors
        Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
        inputs["position_input"] = positionTensor;
        inputs["rotation_input"] = rotationTensor;

        // Execute the model and retrieve the output tensor
        Debug.Log("********** Call classification Execute() **********");
        Tensor outputY = _engine.Execute(inputs).PeekOutput();


        //Debug.Log("Set Prediction");
        prediction.SetPrediction(outputY);
        //Debug.Log("Dispose of Tensor Input manually");
        //inputX.Dispose();

        Debug.Log("Gesture ID is: " + prediction.predictedValue + " with the probability of: " + prediction.probability);

        Debug.Log("********** Save Gesture Data **********");
        SaveGestureData(prediction.predictedValue, prediction.probability);

        Debug.Log("********** Update Detected Gesture ID **********");
        UpdateDetectedGestureID(prediction.predictedValue, prediction.probability);

        // Dispose the tensors
        positionTensor.Dispose();
        rotationTensor.Dispose();
        outputY.Dispose();
        /* End Barracuda */
#endif
    }

    private async void SendTimeToESP()
    {
        Debug.Log("Send Time to ESP32");
        if(SocketClient != null)
        {
            string currentTime = DateTime.Now.ToString("HH:mm:ss");
            await SocketClient.SendMessageAsync(currentTime);
        }
        else
        {
            Debug.Log("SocketClient instance not found");
        }
    }

    public void StartGestureClassification()
    {

        if(!sentFlag)
        {
            sentFlag = true;
            SendTimeToESP();
        }
     
        if (trackedHand == TrackedHandside.Left)
        {
            if (HandTracker.Left.TryLocateHandJoints(FrameTime.OnUpdate, handJointLocations))
            {
                ClassifyGesturesRoutine();
            }
        }
        else
        {
            if (HandTracker.Right.TryLocateHandJoints(FrameTime.OnUpdate, handJointLocations))
            {
                ClassifyGesturesRoutine();
            }
        }
    }

#if !UNITY_EDITOR && UNITY_METRO
    /* Start Barracuda */
    private List<float> LoadTmpData()
    {
        List<float> retData = new List<float>();

        List<float> tmpPosData = new List<float>();
        List<float> tmpRotData = new List<float>();

        List<float> relPosData = new List<float>();
        List<float> relRotData = new List<float>();

        Debug.Log("Load folder: " + TMP_DATA_PATH);
        string jsonData = File.ReadAllText(TMP_DATA_PATH);

        // Parse the JSON string
        Debug.Log("Parse JSON file");
        JObject json = JObject.Parse(jsonData);
        if (json != null)
        {
            GestureData gestureData = new GestureData
            {
                time = json["time"].ToString(),
                Joints = json["Joints"].ToObject<Dictionary<string, JointData>>()
            };

            if (gestureData.Joints != null)
            {
                //Debug.Log("Access values");
                foreach (KeyValuePair<string, JointData> joint in gestureData.Joints)
                {
                    string jointName = joint.Key;
                    JointData jointData = joint.Value;

                    List<float> pos = jointData.Position;
                    List<float> rot = jointData.Rotation;

                    tmpPosData.AddRange(pos);
                    tmpRotData.AddRange(rot);
                }
                //Debug.Log("Convert Pose");
                //relRotData = ConvertGlobRotToRelRot(tmpRotData);
                //relPosData = ConvertGlobPosToRelPos(tmpPosData);

                retData.AddRange(tmpPosData);
                retData.AddRange(tmpRotData);
            }
        }

        return retData;
    }

    private List<float> ConvertGlobPosToRelPos(List<float> pos)
    {
        List<float> tmpPosData = new List<float>();
        float base_x = 0f;
        float base_y = 0f;
        float base_z = 0f;

        //Debug.Log("List size: " + pos.Count);

        for (int index = 0; index < pos.Count; index += 3)
        {
            //Debug.Log("Access values x, y");
            float x = pos[index];
            float y = pos[index + 1];
            float z = pos[index + 2];
            //Debug.Log("X is "+ x);
            //Debug.Log("Y is " + y);
            if (index == 0)
            {
                base_x = x;
                base_y = y;
                base_z = z;
            }

            tmpPosData.Add(x - base_x);
            tmpPosData.Add(y - base_y);
            tmpPosData.Add(z - base_z);
        }

        float max_value = Mathf.Max(tmpPosData.ToArray());
        //Debug.Log("Max Value for normalisation: " + max_value);
        for (int i = 0; i < tmpPosData.Count; i++)
        {
            tmpPosData[i] /= max_value;
        }

        return tmpPosData;
    }

    private List<float> ConvertGlobRotToRelRot(List<float> rot)
    {
        List<float> tmpRotData = new List<float>();
        float base_x = 0f;
        float base_y = 0f;
        float base_z = 0f;

        for (int index = 0; index < rot.Count; index += 3)
        {
            float x = rot[index];
            float y = rot[index + 1];
            float z = rot[index + 2];

            if (index == 0)
            {
                base_x = x;
                base_y = y;
                base_z = z;
            }

            tmpRotData.Add(x - base_x);
            tmpRotData.Add(y - base_y);
            tmpRotData.Add(z - base_z);
        }

        float max_value = Mathf.Max(tmpRotData.ToArray());

        for (int i = 0; i < tmpRotData.Count; i++)
        {
            tmpRotData[i] /= max_value;
        }

        return tmpRotData;
    }

    private void SaveGestureData(int gestureID, float gestureProbability)
    {
        string currentTime = DateTime.Now.ToString("HH:mm:ss");
        string updatedJsonData;

        if(gestureProbability < 0.7)
        {
            gestureID = 0;
        }

        Dictionary<string, string> new_data = new Dictionary<string, string>()
        {
            { "Time", currentTime },
            { "GestureID", gestureID.ToString() },
            { "Probability", gestureProbability.ToString() }
        };


        List<Dictionary<string, string>> existing_data = new List<Dictionary<string, string>>();

        try
        {
            string jsonData = File.ReadAllText(EMG_TIME_LABEL_DATA_PATH);
            existing_data = JsonConvert.DeserializeObject<List<Dictionary<string, string>>>(jsonData);
        }
        catch (FileNotFoundException){}

        existing_data.Add(new_data);
        updatedJsonData = JsonConvert.SerializeObject(existing_data);
        
        File.WriteAllText(EMG_TIME_LABEL_DATA_PATH, updatedJsonData);
    }

    private void UpdateDetectedGestureID(int gestureID, float gestureProbability)
    {
        if (gestureProbability > 0.7f)
        {
            if (gestureID == 0)
            {
                Debug.Log("********** None_Gesture **********");
                detectedGestureID.text = "None";
            }
            else if (gestureID == 1)
            {
                Debug.Log("********** Fist_Gesture **********");
                detectedGestureID.text = "Fist";
            }
            else if (gestureID == 2)
            {
                Debug.Log("********** Pinch_Gesture **********");
                detectedGestureID.text = "Pinch";
            }
            else if (gestureID == 3)
            {
                Debug.Log("********** Spread_Gesture **********");
                detectedGestureID.text = "Spread";
            }
            else if (gestureID == 4)
            {
                Debug.Log("********** Thumb_Up_Gesture **********");
                detectedGestureID.text = "Thumb up";
            }
        }
        else
        {
            detectedGestureID.text = "None";
        }
    }
    /* End Barracuda */
#endif
}
