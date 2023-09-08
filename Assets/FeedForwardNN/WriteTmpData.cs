using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

#if !UNITY_EDITOR && UNITY_METRO
using System.Threading.Tasks;
using Windows.Storage;
#endif

public class WriteTmpData : MonoBehaviour, IMixedRealitySourceStateHandler, IMixedRealityHandJointHandler
{
    int avgNum = 10;


#if UNITY_EDITOR
    string pathToTmpData = "./Assets/FeedForwardNN/data/currentdata.txt";
    string pathToAvgData = "./Assets/FeedForwardNN/data/tmpdata.txt";
#elif !UNITY_EDITOR && UNITY_METRO
    private string TMP_DATA_PATH = "FNN";
    private string tmp_data_file = "FNN/currentdata.txt";
    private string AvgDataFile = "FNN/tmpdata.txt";
#endif
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

    private Dictionary<string, string> HandJointDic;
    private Dictionary<string, string> HandJointRight;

    private MixedRealityPose wristRef;

    private TrackedHandside trackedHand;
    public HandSideManager myCurrHandSide;
    private int FrameIndex;

    //Dictionary<string, JointData> averageRelData;
    GestureData averageRelData;

    void OnEnable()
    {
        Debug.Log("********** OnEnable WriteTmpData.cs **********");
        HandJointDic = new Dictionary<string, string>();
        HandJointRight = new Dictionary<string, string>();
        CoreServices.InputSystem.RegisterHandler<IMixedRealityHandJointHandler>(this);
        FrameIndex = 0;

#if !UNITY_EDITOR && UNITY_METRO
        TMP_DATA_PATH = Path.Combine(ApplicationData.Current.RoamingFolder.Path, TMP_DATA_PATH);
        if (!Directory.Exists(TMP_DATA_PATH))
        {
            Directory.CreateDirectory(TMP_DATA_PATH);
        }
        TMP_DATA_PATH = Path.Combine(ApplicationData.Current.RoamingFolder.Path, tmp_data_file);
        AvgDataFile = Path.Combine(ApplicationData.Current.RoamingFolder.Path, AvgDataFile);
#endif
    }

    public void OnHandJointsUpdated(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData)
    {
        string tmpData;
        trackedHand = myCurrHandSide.GetCurrentHandSide();
        if ((trackedHand == TrackedHandside.Right && eventData.Handedness == Handedness.Right) || (trackedHand == TrackedHandside.Left && eventData.Handedness == Handedness.Left))
        { 
            HandJointDic.Clear();
            var dataTime = DateTime.Now.ToString("HH:mm:ss.ffffff");
            HandJointDic.Add("Time", String.Format("\"{0}\"", dataTime));
            HandJointDic.Add("Joints", DictionaryToString(GetJointPositionDic(eventData, HandJointRight)));
            tmpData = DictionaryToString(HandJointDic);

            WriteAverage("[" + tmpData + "]");

#if UNITY_EDITOR
            File.WriteAllText(pathToTmpData, "[" + tmpData + "]");
#elif !UNITY_EDITOR && UNITY_METRO
            File.WriteAllText(TMP_DATA_PATH, "[" + tmpData + "]");
#endif
            FrameIndex++;
            if(FrameIndex == avgNum+1)
            {
                FrameIndex = 0;
            }
        }
    }

    private void WriteAverage(string data)
    {
        GestureData tmp;
        JArray jsonArray = JArray.Parse(data);
        if (jsonArray.Count > 0)
        {
            JObject json = jsonArray[0].ToObject<JObject>();

            tmp = new GestureData
            {
                time = json["Time"].ToString(),
                Joints = json["Joints"].ToObject<Dictionary<string, JointData>>()
            };

            //Convert tmp Data
            tmp.Joints = ConvertGlobToRel(tmp.Joints);

            if (FrameIndex == 0)
            {
                averageRelData = tmp;
            }
            else
            {
                foreach (KeyValuePair<string, JointData> joint in tmp.Joints)
                {

                    averageRelData.Joints[joint.Key].Position[0] += joint.Value.Position[0];
                    averageRelData.Joints[joint.Key].Position[1] += joint.Value.Position[1];
                    averageRelData.Joints[joint.Key].Position[2] += joint.Value.Position[2];

                    averageRelData.Joints[joint.Key].Rotation[0] += joint.Value.Rotation[0];
                    averageRelData.Joints[joint.Key].Rotation[1] += joint.Value.Rotation[1];
                    averageRelData.Joints[joint.Key].Rotation[2] += joint.Value.Rotation[2];
                }


                if (FrameIndex == avgNum)
                {
                    foreach (KeyValuePair<string, JointData> joint in averageRelData.Joints)
                    {
                        averageRelData.Joints[joint.Key].Position[0] = joint.Value.Position[0] / avgNum;
                        averageRelData.Joints[joint.Key].Position[1] = joint.Value.Position[1] / avgNum;
                        averageRelData.Joints[joint.Key].Position[2] = joint.Value.Position[2] / avgNum;

                        averageRelData.Joints[joint.Key].Rotation[0] = joint.Value.Rotation[0] / avgNum;
                        averageRelData.Joints[joint.Key].Rotation[1] = joint.Value.Rotation[1] / avgNum;
                        averageRelData.Joints[joint.Key].Rotation[2] = joint.Value.Rotation[2] / avgNum;
                    }

                    Dictionary<string, string> dic = new Dictionary<string, string>()
                    {
                        { "Time", tmp.time },
                        { "Joints", JsonConvert.SerializeObject(averageRelData) }
                    };

                    string writeAvgData = JsonConvert.SerializeObject(averageRelData);

#if UNITY_EDITOR
                    File.WriteAllText(pathToAvgData, writeAvgData);
#elif !UNITY_EDITOR && UNITY_METRO
                    File.WriteAllText(AvgDataFile, writeAvgData);
#endif
                }
            }
        }
    }

    private Dictionary<string, JointData> ConvertGlobToRel(Dictionary<string, JointData> glob)
    {
        Dictionary<string, JointData> rel = new Dictionary<string, JointData>();

        float pos_base_x = 0f;
        float pos_base_y = 0f;
        float pos_base_z = 0f;

        float rot_base_x = 0f;
        float rot_base_y = 0f;
        float rot_base_z = 0f;

        float max_pos_value = float.MinValue;
        float max_rot_value = float.MinValue;

        foreach (KeyValuePair<string, JointData> joint in glob)
        {
            List<float> position = joint.Value.Position;
            List<float> rotation = joint.Value.Rotation;

            float pos_x = position[0];
            float pos_y = position[1];
            float pos_z = position[2];

            float rot_x = rotation[0];
            float rot_y = rotation[1];
            float rot_z = rotation[2];

            if (joint.Key == "Wrist")
            {
                pos_base_x = pos_x;
                pos_base_y = pos_y;
                pos_base_z = pos_z;

                rot_base_x = rot_x;
                rot_base_y = rot_y;
                rot_base_z = rot_z;
            }

            position[0] = pos_x - pos_base_x;
            position[1] = pos_y - pos_base_y;
            position[2] = pos_z - pos_base_z;

            rotation[0] = rot_x - rot_base_x;
            rotation[1] = rot_y - rot_base_y;
            rotation[2] = rot_z - rot_base_z;

            // Find the maximum absolute value in the data for normalization
            max_pos_value = Math.Max(max_pos_value, Math.Max(Math.Abs(position[0]), Math.Max(Math.Abs(position[1]), Math.Abs(position[2]))));
            max_rot_value = Math.Max(max_rot_value, Math.Max(Math.Abs(rotation[0]), Math.Max(Math.Abs(rotation[1]), Math.Abs(rotation[2]))));


            JointData tmpJoint = new JointData
            {
                Position = position,
                Rotation = rotation
            };

            rel.Add(joint.Key, tmpJoint);
        }

        foreach (KeyValuePair<string, JointData> joint in rel)
        {
            List<float> position = joint.Value.Position;
            List<float> rotation = joint.Value.Rotation;

            position[0] /= max_pos_value;
            position[1] /= max_pos_value;
            position[2] /= max_pos_value;

            rotation[0] /= max_rot_value;
            rotation[1] /= max_rot_value;
            rotation[2] /= max_rot_value;
        }

        return rel;
    }

    private string DictionaryToString(Dictionary<string, string> dictionary)
    {
        string dictionaryString = "{";

        foreach (KeyValuePair<string, string> keyValues in dictionary)
        {
            dictionaryString += "\"" + keyValues.Key + "\" :" + keyValues.Value + ",";
        }
        dictionaryString = dictionaryString.TrimEnd(' ', ',');
        dictionaryString += "}";
        return dictionaryString;
    }

    // Get position data write to txt file
    private Dictionary<string, string> GetJointPositionDic(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData, Dictionary<string, string> handDicBody)
    {
        handDicBody.Clear();

        //Wrist
        if (eventData.InputData.TryGetValue(TrackedHandJoint.Wrist, out MixedRealityPose wristPose))
        {
            //Debug.Log("Right Wrist POSE:" + wristPose.Position);
            handDicBody.Add("Wrist", JointDescriptionToString(wristPose));
            wristRef = wristPose;
        }

        //Thumb
        if (eventData.InputData.TryGetValue(TrackedHandJoint.ThumbMetacarpalJoint, out MixedRealityPose thumbMetacarpalPose))
        {
            handDicBody.Add("ThumbMetacarpal", JointDescriptionToString(thumbMetacarpalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.ThumbProximalJoint, out MixedRealityPose thumbProximalPose))
        {
            handDicBody.Add("ThumbProximal", JointDescriptionToString(thumbProximalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.ThumbDistalJoint, out MixedRealityPose thumbDistalPose))
        {
            handDicBody.Add("ThumbDistal", JointDescriptionToString(thumbDistalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.ThumbTip, out MixedRealityPose thumbTipPose))
        {
            handDicBody.Add("ThumbTip", JointDescriptionToString(thumbTipPose));
        }

        //Index
        if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexMetacarpal, out MixedRealityPose indexMetacarpalPose))
        {
            handDicBody.Add("IndexMetacarpal", JointDescriptionToString(indexMetacarpalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexKnuckle, out MixedRealityPose indexKnucklePose))
        {
            handDicBody.Add("IndexKnuckle", JointDescriptionToString(indexKnucklePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexMiddleJoint, out MixedRealityPose indexMiddlePose))
        {
            handDicBody.Add("IndexMiddle", JointDescriptionToString(indexMiddlePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexDistalJoint, out MixedRealityPose indexDistalPose))
        {
            handDicBody.Add("IndexDistal", JointDescriptionToString(indexDistalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.IndexTip, out MixedRealityPose indexTipPose))
        {
            handDicBody.Add("IndexTip", JointDescriptionToString(indexTipPose));
        }

        //Middle
        if (eventData.InputData.TryGetValue(TrackedHandJoint.MiddleMetacarpal, out MixedRealityPose middleMetacarpalPose))
        {
            handDicBody.Add("MiddleMetacarpal", JointDescriptionToString(middleMetacarpalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.MiddleKnuckle, out MixedRealityPose middleKnucklePose))
        {
            handDicBody.Add("MiddleKnuckle", JointDescriptionToString(middleKnucklePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.MiddleMiddleJoint, out MixedRealityPose middleMiddlePose))
        {
            handDicBody.Add("MiddleMiddle", JointDescriptionToString(middleMiddlePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.MiddleDistalJoint, out MixedRealityPose middleDistalPose))
        {
            handDicBody.Add("MiddleDistal", JointDescriptionToString(middleDistalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.MiddleTip, out MixedRealityPose middleTipPose))
        {
            handDicBody.Add("MiddleTip", JointDescriptionToString(middleTipPose));
        }

        //Ring
        if (eventData.InputData.TryGetValue(TrackedHandJoint.RingMetacarpal, out MixedRealityPose ringMetacarpalPose))
        {
            handDicBody.Add("RingMetacarpal", JointDescriptionToString(ringMetacarpalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.RingKnuckle, out MixedRealityPose ringKnucklePose))
        {
            handDicBody.Add("RingKnuckle", JointDescriptionToString(ringKnucklePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.RingMiddleJoint, out MixedRealityPose ringMiddlePose))
        {
            handDicBody.Add("RingMiddle", JointDescriptionToString(ringMiddlePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.RingDistalJoint, out MixedRealityPose ringDistalPose))
        {
            handDicBody.Add("RingDistal", JointDescriptionToString(ringDistalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.RingTip, out MixedRealityPose ringTipPose))
        {
            handDicBody.Add("RingTip", JointDescriptionToString(ringTipPose));
        }

        //Pinky
        if (eventData.InputData.TryGetValue(TrackedHandJoint.PinkyMetacarpal, out MixedRealityPose pinkyMetacarpalPose))
        {
            handDicBody.Add("PinkyMetacarpal", JointDescriptionToString(pinkyMetacarpalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.PinkyKnuckle, out MixedRealityPose pinkyKnucklePose))
        {
            handDicBody.Add("PinkyKnuckle", JointDescriptionToString(pinkyKnucklePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.PinkyMiddleJoint, out MixedRealityPose pinkyMiddlePose))
        {
            handDicBody.Add("PinkyMiddle", JointDescriptionToString(pinkyMiddlePose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.PinkyDistalJoint, out MixedRealityPose pinkyDistalPose))
        {
            handDicBody.Add("PinkyDistal", JointDescriptionToString(pinkyDistalPose));
        }
        if (eventData.InputData.TryGetValue(TrackedHandJoint.PinkyTip, out MixedRealityPose pinkyTipPose))
        {
            handDicBody.Add("PinkyTip", JointDescriptionToString(pinkyTipPose));
        }
        return handDicBody;
    }

    // create joint description
    private string JointDescriptionToString(MixedRealityPose fingerPose)
    {
        return String.Format("{{\"Position\" : {0}, \"Rotation\" : {1}}}", PositionArrayToString(fingerPose), RotationArrayToString(fingerPose));
    }

    // convert position data array to json string
    private string PositionArrayToString(MixedRealityPose fingerPose)
    {
        Vector3 localPosition = Quaternion.Inverse(wristRef.Rotation) * (fingerPose.Position - wristRef.Position);
        return String.Format("[{0},{1},{2}]", localPosition[0], localPosition[1], localPosition[2]);
        //return String.Format("[{0},{1},{2}]", fingerPose.Position.x, fingerPose.Position.y, fingerPose.Position.z);
    }

    // convert rotation data array to json string
    private string RotationArrayToString(MixedRealityPose fingerPose)
    {
        Quaternion localRotation = wristRef.Rotation * fingerPose.Rotation;
        localRotation = Quaternion.Euler(localRotation[0], localRotation[1], localRotation[2]);
        return String.Format("[{0},{1},{2}]", localRotation[0], localRotation[1], localRotation[2]);
        //return String.Format("[{0},{1},{2}]", fingerPose.Rotation.x, fingerPose.Rotation.y, fingerPose.Rotation.z);
    }

    public void OnSourceDetected(SourceStateEventData eventData)
    {
        throw new NotImplementedException();
    }

    public void OnSourceLost(SourceStateEventData eventData)
    {
        throw new NotImplementedException();
    }
}
