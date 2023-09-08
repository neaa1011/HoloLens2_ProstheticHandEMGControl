using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine;

#if !UNITY_EDITOR && UNITY_METRO
using System.Threading.Tasks;
using Windows.Storage;
#endif

public class FileIO : MonoBehaviour
{
    public string pinchFolder = "Pinch_Gesture";
    public string fistFolder = "Fist_Gesture";
    public string spreadFolder = "Spread_Gesture";
    public string thumbUpFolder = "Thumb_Up_Gesture";

    public string restFolder = "Rest_Gesture";
    private static Stream OpenFileForWrite(string folderName, string fileName)
    {
        Stream stream = null;
#if !UNITY_EDITOR && UNITY_METRO
        Task<Task> task = Task<Task>.Factory.StartNew(
            async () =>
            {
                StorageFolder folder = await StorageFolder.GetFolderFromPathAsync(folderName);
                StorageFile file = await folder.CreateFileAsync(fileName, CreationCollisionOption.ReplaceExisting);
                stream = await file.OpenStreamForWriteAsync();
            });
        task.Start();
        task.Result.Wait();
#else
        stream = new FileStream(Path.Combine(folderName, fileName), FileMode.Append, FileAccess.Write);
#endif
        return stream;
    }


    public void WriteStringToFile(string s, string fileName, string fileExtension, string gestureFolder)
    {
        string folderPath;

#if !UNITY_EDITOR && UNITY_METRO
        folderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, gestureFolder);

        Task<Task> task = Task<Task>.Factory.StartNew(
            async () =>
            {
                StorageFolder folder = await StorageFolder.GetFolderFromPathAsync(folderPath);
                StorageFile file = await folder.GetFileAsync(String.Format("{0}.{1}", fileName, fileExtension));
                Windows.Storage.FileIO.AppendTextAsync(file, s + "\n");
            });
#else
        folderPath = Path.Combine(Application.persistentDataPath, gestureFolder);

        using (Stream stream = OpenFileForWrite(folderPath, String.Format("{0}.{1}", fileName, fileExtension)))
        {
            byte[] data = Encoding.ASCII.GetBytes(s + "\n");
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
#endif
    }

    public void CreateDataSetDirectory()
    {
        string pinchFolderPath;
        string fistFolderPath;
        string spreadFolderPath;
        string thumbUpFolderPath;
        string restFolderPath;

#if !UNITY_EDITOR && UNITY_METRO
        pinchFolderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, pinchFolder);
        fistFolderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, fistFolder);
        spreadFolderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, spreadFolder);
        thumbUpFolderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, thumbUpFolder);
        restFolderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, restFolder);
#else
        pinchFolderPath = Path.Combine(Application.persistentDataPath, pinchFolder);
        fistFolderPath = Path.Combine(Application.persistentDataPath, fistFolder);
        spreadFolderPath = Path.Combine(Application.persistentDataPath, spreadFolder);
        thumbUpFolderPath = Path.Combine(Application.persistentDataPath, thumbUpFolder);
        restFolderPath = Path.Combine(Application.persistentDataPath, restFolder);
#endif

        if (!Directory.Exists(pinchFolderPath))
        {
            Directory.CreateDirectory(pinchFolderPath);
        }
       
        if (!Directory.Exists(fistFolderPath))
        {
            Directory.CreateDirectory(fistFolderPath);
        }
      
        if (!Directory.Exists(spreadFolderPath))
        {
            Directory.CreateDirectory(spreadFolderPath);
        }
       
        if (!Directory.Exists(thumbUpFolderPath))
        {
            Directory.CreateDirectory(thumbUpFolderPath);
        }
        if (!Directory.Exists(restFolderPath))
        {
            Directory.CreateDirectory(restFolderPath);
        }
    }

    public void CreateDataSetFile(string gestureFolder, string fileName, string fileExtension)
    {
        string folderPath;
        string gesturePath;

#if !UNITY_EDITOR && UNITY_METRO
        folderPath = Path.Combine(ApplicationData.Current.RoamingFolder.Path, gestureFolder);

        Task<Task> task = Task<Task>.Factory.StartNew(
            async () =>
            {
                StorageFolder folder = await StorageFolder.GetFolderFromPathAsync(folderPath);
                StorageFile file = await folder.CreateFileAsync(String.Format("{0}.{1}", fileName, fileExtension), CreationCollisionOption.ReplaceExisting);
            });
#else
        folderPath = Application.persistentDataPath;
        gesturePath = Path.Combine(folderPath, gestureFolder);

        Stream stream = new FileStream(Path.Combine(gesturePath, fileName), FileMode.CreateNew, FileAccess.Write);
        stream.Flush();

#endif
    }

}
