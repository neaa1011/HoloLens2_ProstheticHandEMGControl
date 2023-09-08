using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandednessInstance : Singleton
{
    public TrackedHandside currentHandside = TrackedHandside.Left;

    public void SetHandedness(TrackedHandside myHandedness)
    {
        currentHandside = myHandedness;
    }
}
