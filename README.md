## HoloLens2_ProstheticHandEMGControl

# Overview
This project implements a HoloLens 2 application of a virtual prosthetic hand attached to the residual limb controlled through electromyography. The application provides a feature to create a database of different gestures based on electromyogrphy signals, by instructing the user to perform gestures in a random order, whereas the signals are recorded through a EMG wristband. It is assumed that the subject only has one amputated hand, with the purpose of improving the dataset quality by instructing the user to perform the gesture with both the healthy and amputated hand simultaneously. This is based on the assumption that signal quality increases by simultaneous muscle activation on both sides during a gesture in addition to having visual feedback of the healthy hand. The healthy hand is tracked visually, and the gestures classified on skeleton data. The classification label is then stored alongside the recorded electromyography signal database. However, the application also serves the purpose of providing the user with visual feedback on the performed gesture to act as a training tool. This allows the patient to immediately start rehabilitation without additional costs. Rehabilitation for traumatic limb amputation primarily involves prosthetic training with repetitive tasks and in an advanced state reintegration through daily activities, sports, and work.

# Set up Unity project
This projected was developed in Unity for the HoloLens 2. Therefore, it is necessary to switch the Build Platform to Universal Windows Platform and add the necessary packages manually via the Package Manager, all packages and namespaces are listed in the end of the README file. Additionally, the Windows SDK and the MRTK package need to be installed manually. To import the MRTK package, the execution file is launched and the path to to the project in Unity is defined. The following image marks the required features for this project in the window discover features. This includes the core framework of MRTK to handle input, spatial mapping, spatial awareness, and user interactions. Additionally, pre-built assets and editor tools and utilities are imported. The OpenXR plugin enables running OpenXR in runtime in Unity for developing cross-platform mixed-reality applications.

<img src="https://github.com/neaa1011/HoloLens2_ProstheticHandEMGControl/assets/74051780/a3ee7ba2-00da-499a-923b-7bf4643bdf4f" width="658" height="769">


After importing the necessary features, the OpenXR plugin needs to be selected in the MRTK Configurator in Unity to support the HoloLens 2. In the next step, in the Project Settings under XR Plug-in Management the Universal Windows Platform, Initialise XR on Startup, and  OpenXR and Microsoft HoloLens feature group needs to be selected, marked in the following picture.

![image](https://github.com/neaa1011/HoloLens2_ProstheticHandEMGControl/assets/74051780/ddfd1662-f242-4281-977f-19ded41eaa1c)

By applying the modifications on the project, validation errors may occur, which need to be resolved, except for scene-specific issues.

# Compile Unity project
To compile and debug the XR application on the developer PC, ensure that the Developer Mode on the HoloLens and the PC is enabled. Both can be found under settings under Update and Security. MRTK provides a Build Window to support the process of compiling a XR application to the HoloLens. The first tab, shown in the following image, is to build an internal Unity Project, where the target device is defined, and all scenes are selected which should be included in the build process.

![MRTKBuildTab1](https://github.com/neaa1011/HoloLens2_ProstheticHandEMGControl/assets/74051780/89a1accc-b218-4474-8c9e-28502a5ead58)

The next Image shows the build options for the XR application, also called Appx. The first option is the build configuration which can be selected between Debug, Master and Release. In the Debug configuration all optimizations are off and the profiler is enabled with the purpose to debug the scripts. The Master mode turns on all optimisations and disables the profiler with the purpose to submit the app to the app store. The last configuration mode is Release where all optimizations are turned on and the profiler is enabled to evaluate the app performance. The second important option to select is the Build Platform. Since the Appx is for the HoloLens 2, ARM64 is selected. Build AppX starts then the build process.

![MRTKBuildTab2](https://github.com/neaa1011/HoloLens2_ProstheticHandEMGControl/assets/74051780/8d5f998a-91c7-464c-b360-942a4a8c0cc0)

In the last relevant tab to compile the Appx the deploy options are defined, shown in the next image. Depending on how the HoloLens is connected to the developer PC the target type is selected. If the HoloLens is connected via USB, the Local option is selected. In this project the Remote option was selected. Therefore, a new remote connection needs to be added by entering the local IP address of the HoloLens in the browser of the developer PC. This allows to set the credentials to enable the compilation. However, this process is checked by displaying a Pin in the HoloLens device which needs to be confirmed in the developer PC. The connection can then be tested and if successful the Appx can be installed on the HoloLens 2.

![MRTKBuildTab3](https://github.com/neaa1011/HoloLens2_ProstheticHandEMGControl/assets/74051780/e0a08d6f-7345-4a19-bcff-850931be5303)

# Integration with EMG wristband
The application provides features to control the virtual prosthetic hand through electromyography, and the labelling of the recorded EMG signals through visual gesture classification. However. the labelling currently needs a manual synchronistation with the recorded EMG signals, saved in the file EMGTimeLabel.txt. This file can be accessed through the Windows Device Portal and synchronised based on the time. The video HoloLens_GestureRecognition.mp4 shows the application in combination with the EMG wristband.

# Feedforward Neural Network
The visual gesture recognition is based on a feedforwad neural network (FNN) implemented in Python and exported as onnx model (Assets/FeedForwardNN/VisualGestureRecognition_CreateModel.py). The trained onnx model can be handeled as asset in Unity and using the Barracuda package classifications can be done during runtime. The architecture of the FNN is visualised in the following image, with the relative joint position and rotation as input and four different gesture labels as output.

<img src="https://github.com/neaa1011/HoloLens2_ProstheticHandEMGControl/assets/74051780/f2933156-250c-4eac-a72c-f96a7405b131" width="653" height="359">

# Overview Packages and Namespaces

| Unity Packages | Description |
|----------|----------|
| MRTK 2.8 | Microsoft Mixed Reality Toolkit for Unity provide tools and components for mixed-reality applications. | 
| Python.Net | Package to allow interaction between Unity and Python scripts. |
| OpenXR | API for XR applications. |
| Windows SDK | SDK provides development tools, libraries, and documentation for building Windows applications, including HoloLens 2. |
| Barracuda | Library to run pre-trained neural network models in Unity. |

| C# Namespaces | Description |
|----------|----------|
| UnityEngine | Core Unity functionality, such as Components, Transformations, Physics, Input, and Time. |
| UnityEngine.SceneManagement | Managing scenes in Unity, including loading, unloading, and switching. |
| UnityEditor | Editor API for creating custom Unity Editor scripts and extending the Unity Editor functionality. |
| UnityEditor.Scripting.Python | Integration between Unity and Python scripting within the Unity Editor. |
| Python.Runtime | Running Python scripts within Unity using the Python runtime. |
| Microsoft.MixedReality.OpenXR | Handling OpenXR integration and communication within Unity. |
| Microsoft.MixedReality.Toolkit | Framework for building mixed reality applications with Unity, offering various features and utilities. |
| Microsoft.MixedReality.Toolkit.Input | Handling input interactions, such as hand tracking and gestures, in mixed reality applications. |
| Microsoft.MixedReality.Toolkit.Utilities | Containing utility classes and functions for mixed reality development. |
| System | Fundamental system-related classes and functionalities. |
| System.Collections.Generic | Generic collection classes include lists, dictionaries, and queues. |
| System.IO | Classes for file input/output operations, allowing reading and writing files. |
| System.Text | Provides encoding and decoding functionality for various text formats. |
| System.Threading.Tasks | Interface for asynchronous and parallel programming. |
| System.Net | Allows managing network connections and communication. |
| Newtonsoft.Json | Library for working with JSON data. |
| TMPro | Package for advanced text rendering and typography in Unity. |
| Windows.Storage | Provides access to files, folders, and application data on Universal Windows Platforms (UWP). |

| Python Packages | Description |
|----------|----------|
| Pytorch | Deep learning framework for training and deploying neural networks. |
| Torchvision | PyTorch package that provides datasets, models, and transformations for computer vision tasks. |
| Tensorflow | Deep learning framework for developing and deploying machine learning models. |
| Scikit-learn | Machine learning library for data mining and analysis. |
| Pandas | Data manipulation and analysis library. |
| Seaborn | Statistical data visualization library built on top of Matplotlib. |
| sklearn | Module within Scikit-learn that provides additional tools and utilities. |
| pylance | Language server extension for Python that offers enhanced language features in development environments. |
| copy | Functions for creating copies of objects. |
| os | Module for interacting with the operating system, allowing file and directory operations. |
| glob | Pattern matching functionality for file and directory names. |
| json | Functions for working with JSON data. |
| numpy | Fundamental package for scientific computing in Python. |
| Unity Engine | Access from Python to core Unity functionality, such as Components, Transformations, Physics, Input, and Time. |
| TMPro | Access from Python to text rendering and typography directly in Unity. |
| matplotlib | Interface for a wide range of different plots and visualizations. |
| tf2onnx | Package to convert TensorFlow models to onnx models. |
| onnx | Open Neural Network Exchange to allow the usage of deep learning models on other frameworks like Unity. |
| socket | Package to manage network connections and communication via sockets over TCP/UDP protocols. |




