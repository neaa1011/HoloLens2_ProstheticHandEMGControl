using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UIDynamicFollowCamera : MonoBehaviour
{
    // Reference to the camera transform
    public Transform cameraTransform;

    private void Start()
    {
        CoreServices.DiagnosticsSystem.ShowDiagnostics = false;

        CoreServices.DiagnosticsSystem.ShowProfiler = false;
    }

    // Update is called once per frame
    void Update()
    {
        // Update the position of the Canvas to match the camera position
        transform.position = cameraTransform.position;
        transform.rotation = cameraTransform.rotation;
    }
}
