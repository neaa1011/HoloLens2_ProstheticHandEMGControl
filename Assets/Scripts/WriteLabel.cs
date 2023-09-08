using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

#if !UNITY_EDITOR && UNITY_METRO
using System.Threading.Tasks;
using Windows.Storage;
#endif

public class WriteLabel : MonoBehaviour
{
    [SerializeField] public bool isRandom;


    private string Labelpath = "./Assets/Scripts/data/Label.txt";
    private string file_path;// = "FNN/Label.txt";
    private string Fnn_Path;// = "FNN";


    void Start()
    {
        initPaths();
        if (isRandom)
        {
            InvokeRepeating("WriteRandomLabelInFile", 2f, 4f); //Start after 2 sec, called every 4 sec
        }
    }

    public void initPaths()
    {
        file_path = "FNN/Label.txt";
        Fnn_Path = "FNN";
#if !UNITY_EDITOR && UNITY_METRO
        Fnn_Path = Path.Combine(ApplicationData.Current.RoamingFolder.Path, Fnn_Path);
        if (!Directory.Exists(Fnn_Path))
        {
            Directory.CreateDirectory(Fnn_Path);
        }
        Fnn_Path = Path.Combine(ApplicationData.Current.RoamingFolder.Path, file_path);
#endif
    }

    private void WriteRandomLabelInFile()
    {
        int randomNumber = Random.Range(0, 5);
        Debug.Log($"********** Write Random Label {randomNumber}**********");
#if UNITY_EDITOR
        File.WriteAllText(Labelpath, "[" + "{\"GestureID\":\"" + randomNumber + "\"}]");
#elif !UNITY_EDITOR && UNITY_METRO
        File.WriteAllText(Fnn_Path, "[" + "{\"GestureID\":\"" + randomNumber + "\"}]");
#endif
    }

    public void WriteLabelInFile(int label)
    {
#if UNITY_EDITOR
        File.WriteAllText(Labelpath, "[" + "{\"GestureID\":\"" + label + "\"}]");
#elif !UNITY_EDITOR && UNITY_METRO
        File.WriteAllText(Fnn_Path, "[" + "{\"GestureID\":\"" + label + "\"}]");
#endif
    }
}
