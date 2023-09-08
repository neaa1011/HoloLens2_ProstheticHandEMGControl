using Microsoft.MixedReality.OpenXR;
using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.IO;
using TMPro;
using System.Collections;
using System;
using Newtonsoft.Json.Linq;
using UnityEngine.SceneManagement;

#if !UNITY_EDITOR && UNITY_METRO
using System.Threading.Tasks;
using Windows.Storage;
#endif


public class RightHandController : MonoBehaviour
{
    public  string GestureID;
    [SerializeField] private GameObject handObj;
    [SerializeField] private HandProxy handProxy;
    private readonly HandJointLocation[] handJointLocations = new HandJointLocation[HandTracker.JointCount];

    public float speed = 5f;
    private float interpol = 0f;

    private CustomGestureID nextGestureID;
    [SerializeField] public GameObject classifiedGestureTxt;

    private Dictionary<string, Quaternion> RotFistData;
    private Dictionary<string, Vector3> PosFistData;

    public Dictionary<string, Quaternion> RotSpreadData;
    private Dictionary<string, Vector3> PosSpreadData;

    private Dictionary<string, Quaternion> RotPinchData;
    private Dictionary<string, Vector3> PosPinchData;

    private Dictionary<string, Quaternion> RotThumbData;
    private Dictionary<string, Vector3> PosThumbData;

    [SerializeField] public bool isClassifying;
    [SerializeField] public bool isEMG;

    private TrackedHandside isTracked;
    public HandSideManager myCurrHandSide;

    private string LabelPath;

    private void Start()
    {                    
#if UNITY_EDITOR
            LabelPath = "./Assets/Scripts/data/Label.txt";
#elif !UNITY_EDITOR && UNITY_METRO
            LabelPath = "FNN/Label.txt";
            LabelPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, LabelPath);
#endif

        handObj.SetActive(false);

        isTracked = myCurrHandSide.GetCurrentHandSide();
        Debug.Log("********** Tracked hand: " + isTracked.ToString());

        if (isClassifying && (isTracked == TrackedHandside.Left))
        {
            nextGestureID = CustomGestureID.None;

            //Debug.Log("********** Load Files **********");

            RotFistData = new Dictionary<string, Quaternion>();
            PosFistData = new Dictionary<string, Vector3>();

            RotSpreadData = new Dictionary<string, Quaternion>();
            PosSpreadData = new Dictionary<string, Vector3>();

            RotPinchData = new Dictionary<string, Quaternion>();
            PosPinchData = new Dictionary<string, Vector3>();

            RotThumbData = new Dictionary<string, Quaternion>();
            PosThumbData = new Dictionary<string, Vector3>();
#if UNITY_EDITOR
            LoadData(CustomGestureID.Fist, "./Assets/Scripts/data/RightHand/fist.txt");
            LoadData(CustomGestureID.Pinch, "./Assets/Scripts/data/RightHand/pinch.txt");
            LoadData(CustomGestureID.Spread, "./Assets/Scripts/data/RightHand/spread.txt");
            LoadData(CustomGestureID.ThumbsUp, "./Assets/Scripts/data/RightHand/thumbup.txt");
#elif !UNITY_EDITOR && UNITY_METRO
            nextGestureID = CustomGestureID.Fist;
            LoadData(nextGestureID, "RightHand/fist");
            nextGestureID = CustomGestureID.Pinch;
            LoadData(CustomGestureID.Pinch, "RightHand/pinch");
            nextGestureID = CustomGestureID.Spread;
            LoadData(CustomGestureID.Spread, "RightHand/spread");
            nextGestureID = CustomGestureID.ThumbsUp;
            LoadData(CustomGestureID.ThumbsUp, "RightHand/thumbup");
            nextGestureID = CustomGestureID.None;
#endif
            if (isEMG)
            {
                Debug.Log("*********** Set Invoke Repeating GetEMGGstures() **********");
                InvokeRepeating("GetEMGGesture", 2f, 0.25f); //Start after 2 sec, called every 3 sec
            }
            else
            {
                Debug.Log("*********** Set Invoke Repeating GetGestures() **********");
                InvokeRepeating("GetGesture", 2f, 0.25f);
            }  
        }
    }

    private void Update()
    {            
        if (HandTracker.Right.TryLocateHandJoints(FrameTime.OnUpdate, handJointLocations))
        {
            // Enable the hand
            handObj.SetActive(true);
            if (isClassifying)
            {
                isTracked = myCurrHandSide.GetCurrentHandSide();
                if (isTracked == TrackedHandside.Right)
                {
                    // Normal hand tracking
                    UpdateHandJoints();
                }
                else
                {
                    // Prosthetic hand movement
                    SetSpawnPoint();
                    interpol = speed * Time.deltaTime;
                    PerformGesture(interpol);
                }
            }
            else
            {
                // Normal hand tracking
                UpdateHandJoints();
            }
        }
        else
        {
            // Disable the hand
            handObj.SetActive(false);
        }
    }

    private void LoadData(CustomGestureID id, string pPath)
    {
        Debug.Log("*********** Load Data **********");

#if !UNITY_EDITOR && UNITY_METRO
        //pPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, pPath);
        TextAsset textAsset = Resources.Load<TextAsset>(pPath);
        string data = textAsset.text;
#else
        Debug.Log($"*********** Load Data {pPath}");
        string data = File.ReadAllText(pPath);
        //Debug.Log(data);
#endif
        if(data == null)
        {
            Debug.Log("*********** data does not contain data **********");
        }

        List<GestureData> DataList = JsonConvert.DeserializeObject<List<GestureData>>(data);

        if(DataList[0].Joints == null)
        {
            Debug.Log("*********** DataList does not contain data **********");
        }

        switch (id)
        {
            case CustomGestureID.Fist:
                ConvertFloatToTransform(DataList[0].Joints, id);
                break;
            case CustomGestureID.Pinch:
                ConvertFloatToTransform(DataList[0].Joints, id);
                break;
            case CustomGestureID.Spread:
                ConvertFloatToTransform(DataList[0].Joints, id);
                break;
            case CustomGestureID.ThumbsUp:
                ConvertFloatToTransform(DataList[0].Joints, id);
                break;
        }
    }

    private void ConvertFloatToTransform(Dictionary<string, JointData> data, CustomGestureID id)
    {
        Vector3 pos;
        Quaternion rot;

        Debug.Log("*********** ConvertFloatToTransform() **********");

        foreach (KeyValuePair<string, JointData> joint in data)
        {
            JointData tmp = joint.Value;
            List<float> tmp_pos = tmp.Position;
            List<float> tmp_rot = tmp.Rotation;

            pos = new Vector3(tmp_pos[0], tmp_pos[1], tmp_pos[2]);
            rot = new Quaternion(tmp_rot[0], tmp_rot[1], tmp_rot[2], 0f);

            switch (id)
            {
                case CustomGestureID.Fist:
                    RotFistData.Add(joint.Key, rot);
                    PosFistData.Add(joint.Key, pos);
                    break;
                case CustomGestureID.Pinch:
                    RotPinchData.Add(joint.Key, rot);
                    PosPinchData.Add(joint.Key, pos);
                    break;
                case CustomGestureID.Spread:
                    RotSpreadData.Add(joint.Key, rot);
                    PosSpreadData.Add(joint.Key, pos);
                    break;
                case CustomGestureID.ThumbsUp:
                    RotThumbData.Add(joint.Key, rot);
                    PosThumbData.Add(joint.Key, pos);
                    break;
            }
        }
    }

    private void UpdateHandJoints()
    {
        ApplyWrist(handProxy.Wrist, HandJoint.Wrist);
        ApplyJoints(handProxy.ThumbMetacarpal, HandJoint.ThumbMetacarpal);
        ApplyJoints(handProxy.ThumbProximal, HandJoint.ThumbProximal);
        ApplyJoints(handProxy.ThumbDistal, HandJoint.ThumbDistal);
        ApplyJoints(handProxy.ThumbTip, HandJoint.ThumbTip);
        ApplyJoints(handProxy.IndexMetacarpal, HandJoint.IndexMetacarpal);
        ApplyJoints(handProxy.IndexProximal, HandJoint.IndexProximal);
        ApplyJoints(handProxy.IndexIntermediate, HandJoint.IndexIntermediate);
        ApplyJoints(handProxy.IndexDistal, HandJoint.IndexDistal);
        ApplyJoints(handProxy.IndexTip, HandJoint.IndexTip);
        ApplyJoints(handProxy.MiddleMetacarpal, HandJoint.MiddleMetacarpal);
        ApplyJoints(handProxy.MiddleProximal, HandJoint.MiddleProximal);
        ApplyJoints(handProxy.MiddleIntermediate, HandJoint.MiddleIntermediate);
        ApplyJoints(handProxy.MiddleDistal, HandJoint.MiddleDistal);
        ApplyJoints(handProxy.MiddleTip, HandJoint.MiddleTip);
        ApplyJoints(handProxy.RingMetacarpal, HandJoint.RingMetacarpal);
        ApplyJoints(handProxy.RingProximal, HandJoint.RingProximal);
        ApplyJoints(handProxy.RingIntermediate, HandJoint.RingIntermediate);
        ApplyJoints(handProxy.RingDistal, HandJoint.RingDistal);
        ApplyJoints(handProxy.RingTip, HandJoint.RingTip);
        ApplyJoints(handProxy.LittleMetacarpal, HandJoint.LittleMetacarpal);
        ApplyJoints(handProxy.LittleProximal, HandJoint.LittleProximal);
        ApplyJoints(handProxy.LittleIntermediate, HandJoint.LittleIntermediate);
        ApplyJoints(handProxy.LittleDistal, HandJoint.LittleDistal);
        ApplyJoints(handProxy.LittleTip, HandJoint.LittleTip);
    }

    private void SetSpawnPoint()
    {
        ApplyWrist(handProxy.Wrist, HandJoint.Wrist);
    }


    private void GetEMGGesture()
    {
        Debug.Log("*********** Invoke GetEMGGetsure() **********");
        if (isTracked == TrackedHandside.Left)
        {
            Debug.Log($"*********** Load EMG Getsure Label {LabelPath}");
            string data = File.ReadAllText(LabelPath);

            Debug.Log("Load Data from JSON file");
            JArray jsonArray = JArray.Parse(data);
            JObject json = jsonArray[0].ToObject<JObject>();
            GestureID = json["GestureID"].ToString();

            Debug.Log($"********** GetEMGGesture() Label from file is {GestureID}**********");

            nextGestureID = (CustomGestureID)Enum.Parse(typeof(CustomGestureID), GestureID);
        }
    }

    private void GetGesture()
    {
        if (isTracked == TrackedHandside.Left)
        {
            Debug.Log("*********** Get Getsure Label **********");
            TextMeshPro myText = classifiedGestureTxt.GetComponent<TextMeshPro>();
            if(myText == null)
                Debug.Log("FAILED Get Gesture Label");
            switch (myText.text)
            {
                case "Fist":
                    nextGestureID = CustomGestureID.Fist;
                    break;
                case "Pinch":
                    nextGestureID = CustomGestureID.Pinch;
                    break;
                case "Spread":
                    nextGestureID = CustomGestureID.Spread;
                    break;
                case "Thumb up":
                    nextGestureID = CustomGestureID.ThumbsUp;
                    break;
                default:
                    nextGestureID = CustomGestureID.None;
                    break;
            }
        }
    }
    private void PerformGesture(float interpol)
    {
        switch (nextGestureID)
        {
            case CustomGestureID.Fist:
                PerformRotationMovement(RotFistData, interpol);
                break;
            case CustomGestureID.Pinch:
                PerformRotationMovement(RotPinchData, interpol);
                break;
            case CustomGestureID.Spread:
                PerformRotationMovement(RotSpreadData, interpol);
                break;
            case CustomGestureID.ThumbsUp:
                PerformRotationMovement(RotThumbData, interpol);
                break;
            default:
                // Do nothing
                break;
        }
    }

    private void PerformRotationMovement(Dictionary<string, Quaternion> rotDic, float interpol)
    {
        Apply(handProxy.ThumbMetacarpal, rotDic["ThumbMetacarpal"], interpol);
        Apply(handProxy.ThumbProximal, rotDic["ThumbProximal"], interpol);
        Apply(handProxy.ThumbDistal, rotDic["ThumbDistal"], interpol);
        Apply(handProxy.ThumbTip, rotDic["ThumbTip"], interpol);

        Apply(handProxy.IndexMetacarpal, rotDic["IndexMetacarpal"], interpol);
        Apply(handProxy.IndexProximal, rotDic["IndexKnuckle"], interpol);
        Apply(handProxy.IndexIntermediate, rotDic["IndexMiddle"], interpol);
        Apply(handProxy.IndexDistal, rotDic["IndexDistal"], interpol);
        Apply(handProxy.IndexTip, rotDic["IndexTip"], interpol);

        Apply(handProxy.MiddleMetacarpal, rotDic["MiddleMetacarpal"], interpol);
        Apply(handProxy.MiddleProximal, rotDic["MiddleKnuckle"], interpol);
        Apply(handProxy.MiddleIntermediate, rotDic["MiddleMiddle"], interpol);
        Apply(handProxy.MiddleDistal, rotDic["MiddleDistal"], interpol);
        Apply(handProxy.MiddleTip, rotDic["MiddleTip"], interpol);

        Apply(handProxy.RingMetacarpal, rotDic["RingMetacarpal"], interpol);
        Apply(handProxy.RingProximal, rotDic["RingKnuckle"], interpol);
        Apply(handProxy.RingIntermediate, rotDic["RingMiddle"], interpol);
        Apply(handProxy.RingDistal, rotDic["RingDistal"], interpol);
        Apply(handProxy.RingTip, rotDic["RingTip"], interpol);

        Apply(handProxy.LittleMetacarpal, rotDic["PinkyMetacarpal"], interpol);
        Apply(handProxy.LittleProximal, rotDic["PinkyKnuckle"], interpol);
        Apply(handProxy.LittleIntermediate, rotDic["PinkyMiddle"], interpol);
        Apply(handProxy.LittleDistal, rotDic["PinkyDistal"], interpol);
        Apply(handProxy.LittleTip, rotDic["PinkyTip"], interpol);
    }

    private void ApplyWrist(Transform jointTransform, HandJoint joint)
    {
        if (jointTransform == null)
            return;
        HandJointLocation location = handJointLocations[(int)joint];
        Quaternion rot = location.Pose.rotation * Reorientation(handProxy);
        Vector3 pos = location.Pose.position;
        jointTransform.SetPositionAndRotation(pos, rot);
    }

    private void ApplyJoints(Transform jointTransform, HandJoint joint)
    {
        if (jointTransform == null)
            return;
        HandJointLocation location = handJointLocations[(int)joint];
        Quaternion rot = location.Pose.rotation * Reorientation(handProxy);
        jointTransform.rotation = rot;
    }

    private void Apply(Transform jointTransform, Quaternion rot, float interpol)
    {
        if (jointTransform == null)
            return;

        Quaternion targetRotation = Quaternion.Euler(rot[0], rot[1], rot[2]);
        //jointTransform.localRotation = targetRotation;
        jointTransform.localRotation = Quaternion.Slerp(jointTransform.localRotation, targetRotation, interpol);
    }

    private Quaternion Reorientation(HandProxy hand)
    {
        return Quaternion.Inverse(Quaternion.LookRotation(hand.ModelFingerPointing, -hand.ModelPalmFacing));
    }
}
