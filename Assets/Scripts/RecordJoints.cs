
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System;
using System.Collections.Generic;
using UnityEngine;

public class RecordJoints : MonoBehaviour, IMixedRealitySourceStateHandler, IMixedRealityHandJointHandler
{
    private Dictionary<string, string> HandJointDic;    //outermost
    private Dictionary<string, string> HandJoints;  // hand joints

    private FileIO fileio;
    private string JointPositionFileName;

    //position data serial number
    private int FrameSN;

    //Trigger Box
    private MixedRealityPose RightPalmPose;
    private MixedRealityPose LeftPalmPose;

    [SerializeField] public GameObject pinchTriggerObject;
    [SerializeField] public GameObject fistTriggerObject;
    [SerializeField] public GameObject spreadTriggerObject;
    [SerializeField] public GameObject thumbUpTriggerObject;

    [SerializeField] public GameObject restTriggerObject;

    [SerializeField] public bool isLeftHand;

    private float distance = 0;

    private enum Recording_State
    {
        NOT_RECORDING,
        RECORDING_PINCH,
        RECORDING_FIST,
        RECORDING_SPREAD,
        RECORDING_THUMB_UP,
        RECORDING_REST
    }
    private string currentGestureFolder;

    Recording_State currentState;

    // Start is called before the first frame update
    void Start()
    {
        HandJointDic = new Dictionary<string, string>();
        HandJoints = new Dictionary<string, string>();

        fileio = new FileIO();
        fileio.CreateDataSetDirectory();

        currentState = Recording_State.NOT_RECORDING;
    }

    // Update is called once per frame
    void Update()
    {
        if(currentState == Recording_State.NOT_RECORDING)
        {
            if(startRecording())
            {
                Debug.Log("********** Start Recording **********");
                initRecording();
            }
        }
        else
        {
            if(!checkHandPosition())
            {
                disableRecording();
            }          
        }
    }

    public void OnHandJointsUpdated(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData)
    {
        string tmpData;
        var dataTime = DateTime.Now.ToString("HH:mm:ss.ffffff");

        HandJointDic.Clear();
        //Add timestamp to dictionsry
        HandJointDic.Add("Time", String.Format("\"{0}\"", dataTime));

        var camera = Camera.main;
        HandJointDic.Add("MainCamera", String.Format("[{0},{1},{2}]", camera.transform.position.x, camera.transform.position.y, camera.transform.position.z));
        //HandJointDic.Add("Frame", FrameSN.ToString());

        HandJointDic.Add("Gesture", String.Format("\"{0}\"", currentGestureFolder));

        //add right hand data dictionary
        HandJointDic.Add("Joints", DictionaryToString(GetJointPositionDic(eventData, HandJoints)));

        tmpData = DictionaryToString(HandJointDic);
        //Debug.Log("This Data:" + tmpData);
        if (FrameSN == 0)
        {
            fileio.WriteStringToFile("[" + tmpData, JointPositionFileName, "txt", currentGestureFolder);
        }
        else
        {
            fileio.WriteStringToFile("," + tmpData, JointPositionFileName, "txt", currentGestureFolder);
        }
        FrameSN++;
    }

    private bool checkHandPosition()
    {
        if (isLeftHand)
        {
            if ( HandJointUtils.TryGetJointPose(TrackedHandJoint.Palm, Handedness.Left, out LeftPalmPose))
            {
                switch (currentState)
                {
                    case Recording_State.RECORDING_PINCH:
                        if (Vector3.Distance(LeftPalmPose.Position, pinchTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_FIST:
                        if (Vector3.Distance(LeftPalmPose.Position, fistTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_SPREAD:
                        if (Vector3.Distance(LeftPalmPose.Position, spreadTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_THUMB_UP:
                        if (Vector3.Distance(LeftPalmPose.Position, thumbUpTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_REST:
                        if (Vector3.Distance(LeftPalmPose.Position, restTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                }
                return true;
            }
            
        }
        else
        {
            if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Palm, Handedness.Right, out RightPalmPose))
            {
                switch (currentState)
                {
                    case Recording_State.RECORDING_PINCH:
                        if (Vector3.Distance(RightPalmPose.Position, pinchTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_FIST:
                        if (Vector3.Distance(RightPalmPose.Position, fistTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_SPREAD:
                        if (Vector3.Distance(RightPalmPose.Position, spreadTriggerObject.transform.position) > 0.1)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_THUMB_UP:
                        if (Vector3.Distance(RightPalmPose.Position, thumbUpTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                    case Recording_State.RECORDING_REST:
                        if (Vector3.Distance(RightPalmPose.Position, restTriggerObject.transform.position) > 0.1f)
                        {
                            return false;
                        }
                        break;
                }
                return true;
            }          
        }
        return false;
    }

    private bool startRecording()
    {
        if (isLeftHand)
        {
            if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Palm, Handedness.Left, out LeftPalmPose))
            {
                distance = Vector3.Distance(LeftPalmPose.Position, pinchTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Left Hand: Set Recording state to Record Pinch *****");
                    currentGestureFolder = fileio.pinchFolder;
                    currentState = Recording_State.RECORDING_PINCH;
                    return true;
                }
                distance = Vector3.Distance(LeftPalmPose.Position, fistTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Left Hand: Set Recording state to Record Fist *****");
                    currentGestureFolder = fileio.fistFolder;
                    currentState = Recording_State.RECORDING_FIST;
                    return true;
                }
                distance = Vector3.Distance(LeftPalmPose.Position, spreadTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Left Hand: Set Recording state to Record Spread *****");
                    currentGestureFolder = fileio.spreadFolder;
                    currentState = Recording_State.RECORDING_SPREAD;
                    return true;
                }
                distance = Vector3.Distance(LeftPalmPose.Position, thumbUpTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Left Hand: Set Recording state to Record Thumb Up *****");
                    currentGestureFolder = fileio.thumbUpFolder;
                    currentState = Recording_State.RECORDING_THUMB_UP;
                    return true;
                }
                //distance = Vector3.Distance(LeftPalmPose.Position, restTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Left Hand: Set Recording state to Record Thumb Up *****");
                    //currentGestureFolder = fileio.restFolder;
                    //currentState = Recording_State.RECORDING_REST;
                    return true;
                }
            }
        }
        else
        {
            if (HandJointUtils.TryGetJointPose(TrackedHandJoint.Palm, Handedness.Right, out RightPalmPose))
            {
                distance = Vector3.Distance(RightPalmPose.Position, pinchTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Right Hand: Set Recording state to Record Pinch *****");
                    currentGestureFolder = fileio.pinchFolder;
                    currentState = Recording_State.RECORDING_PINCH;
                    return true;
                }
                distance = Vector3.Distance(RightPalmPose.Position, fistTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Right Hand: Set Recording state to Record Fist *****");
                    currentGestureFolder = fileio.fistFolder;
                    currentState = Recording_State.RECORDING_FIST;
                    return true;
                }
                distance = Vector3.Distance(RightPalmPose.Position, spreadTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Right Hand: Set Recording state to Record Spread *****");
                    currentGestureFolder = fileio.spreadFolder;
                    currentState = Recording_State.RECORDING_SPREAD;
                    return true;
                }
                distance = Vector3.Distance(RightPalmPose.Position, thumbUpTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Right Hand: Set Recording state to Record Thumb Up *****");
                    currentGestureFolder = fileio.thumbUpFolder;
                    currentState = Recording_State.RECORDING_THUMB_UP;
                    return true;
                }
                //distance = Vector3.Distance(RightPalmPose.Position, restTriggerObject.transform.position);
                if (distance < 0.1f)
                {
                    //Debug.Log("***** Right Hand: Set Recording state to Record Thumb Up *****");
                    //currentGestureFolder = fileio.restFolder;
                    //currentState = Recording_State.RECORDING_REST;
                    return true;
                }
            }
        }
        return false;
    }

    private void initRecording()
    {
        toggleGestureObjectActiveState(false);
        //Start to track position data
        FrameSN = 0;
        JointPositionFileName = DateTime.Now.ToString("yyyy_MM_dd_T_HH_mm_ss");
        Debug.Log("NEW FILE NAME:" + JointPositionFileName);
        fileio.CreateDataSetFile(currentGestureFolder, JointPositionFileName, "txt");
        CoreServices.InputSystem.RegisterHandler<IMixedRealityHandJointHandler>(this);
        //Debug.Log("ENABLE Recording");
    }

    private void disableRecording()
    {
        fileio.WriteStringToFile("]" , JointPositionFileName, "txt", currentGestureFolder);
        toggleGestureObjectActiveState(true);
        //Stop to track position data
        CoreServices.InputSystem.UnregisterHandler<IMixedRealityHandJointHandler>(this);
        Debug.Log("DISABLE Recording");
        currentState = Recording_State.NOT_RECORDING;
    }

    private void toggleGestureObjectActiveState(bool status)
    {
        pinchTriggerObject.SetActive(status);
        fistTriggerObject.SetActive(status);
        spreadTriggerObject.SetActive(status);
        thumbUpTriggerObject.SetActive(status);
        //restTriggerObject.SetActive(status);
    }

    // Get position data write to txt file
    private Dictionary<string, string> GetJointPositionDic(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData, Dictionary<string, string> handDicBody)
    {
        handDicBody.Clear();

        //Wrist
        if (eventData.InputData.TryGetValue(TrackedHandJoint.Wrist, out MixedRealityPose wristPose))
        {
            handDicBody.Add("Wrist", JointDescriptionToString(wristPose));
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
        return String.Format("[{0},{1},{2}]", fingerPose.Position.x, fingerPose.Position.y, fingerPose.Position.z);
    }

    // convert rotation data array to json string
    private string RotationArrayToString(MixedRealityPose fingerPose)
    {
        return String.Format("[{0},{1},{2}]", fingerPose.Rotation.x, fingerPose.Rotation.y, fingerPose.Rotation.z);
    }

    // convert <string, string> dictionary to json string
    public string DictionaryToString(Dictionary<string, string> dictionary)
    {
        string dictionaryString = "{";
 
        foreach (KeyValuePair<string, string> keyValues in dictionary)
        {
            dictionaryString += "\"" + keyValues.Key + "\" :" + keyValues.Value +",";
        }
        dictionaryString = dictionaryString.TrimEnd(' ', ',');
        dictionaryString += "}";
        return dictionaryString;
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
