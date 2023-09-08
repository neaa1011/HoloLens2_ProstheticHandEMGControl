using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum TrackedHandside
{
    Right,
    Left
}


public class HandSideManager : MonoBehaviour
{
    public HandednessInstance myHandedness;
    private void Start()
    {
        if(HandednessInstance.Instance == null)
        {
            Debug.Log("Get Hand side from singleton not succeded! Set default side to right.");
        }
        else
        {
            myHandedness = (HandednessInstance.Instance as HandednessInstance);
            Debug.Log("Get Hand side from singleton: " + myHandedness.currentHandside.ToString());
        }
    }

    public void toggleHandSide()
    {
        switch(myHandedness.currentHandside)
        {
            case TrackedHandside.Right:
                Debug.Log("Toggle tracked Hand side from right to left");
                myHandedness.SetHandedness(TrackedHandside.Left);
                break;
            case TrackedHandside.Left:
                Debug.Log("Toggle tracked Hand side from left to right");
                myHandedness.SetHandedness(TrackedHandside.Right);
                break;
        }
    }

    public TrackedHandside GetCurrentHandSide()
    {
        return myHandedness.currentHandside;
    }
}
