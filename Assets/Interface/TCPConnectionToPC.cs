
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class TCPConnectionToPC : MonoBehaviour
{
    private const string serverIP = "192.168.4.2";
    private const string hostname = "pcclassification";
    private const int serverPort = 9000;

    private TcpClient client;
    private NetworkStream stream;
    private bool isListening;

    [SerializeField] public WriteLabel writeLabel;

    private async void Start()
    {
        isListening = false;
        writeLabel.initPaths();
        await ConnectToServerAsync();
    }

    private async Task ConnectToServerAsync()
    {
        try
        {
            //IPAddress[] addresses = Dns.GetHostAddresses(hostname);
            //IPAddress ServerIP = addresses[0];
            //Debug.Log($"IP Address: {ServerIP}");
            client = new TcpClient();
            await client.ConnectAsync(serverIP, serverPort);
            //await client.ConnectAsync(ServerIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to server");

            // Start listening for incoming data
            isListening = true;
        }
        catch (Exception e)
        {
            Debug.Log($"Error connecting to server: {e.Message}");
        }
    }

    void Update()
    {
        if (isListening && client != null && client.Connected)
        {
            Debug.Log("Listen for data");
            if (stream.DataAvailable)
            {
                // Perform the network operation asynchronously using Task.Run()               
                Task.Run(async () =>
                {
                    isListening = false;
                    try
                    {
                        Debug.Log("Receiving data async...");
                        byte[] buffer = new byte[client.ReceiveBufferSize];
                        //byte[] buffer = new byte[4];
                        int bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
                        if (bytesRead > 0)
                        {
                            Debug.Log("Bytes: " + bytesRead.ToString());
                            Debug.Log("Buffer: " + buffer);
                            byte[] data = new byte[bytesRead];
                            Array.Copy(buffer, data, bytesRead);
                            string message = Encoding.UTF8.GetString(data);
                            Debug.Log($"Received message: {message}");

                            string[] parts = message.Split(new char[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);

                            // Process the received message
                            //if (int.TryParse(message, out int receivedInt))
                            if (parts.Length > 0 && int.TryParse(parts[0], out int receivedInt))
                            {
                                Debug.Log($"Received integer: {receivedInt}");
                                writeLabel.WriteLabelInFile(receivedInt);
                            }
                            else
                            {
                                Debug.Log($"Message could not be read!");
                            }
                            /*
                            int receivedInt = BitConverter.ToInt32(buffer, 0);
                            Debug.Log($"Received integer: {receivedInt}");                           
                            // Process the received integer
                            //writeLabel.WriteLabelInFile(receivedInt);
                            */
                        }
                    }
                    catch (Exception e)
                    {
                        // Handle the exception here (e.g., log the error or display a message)
                        Debug.Log($"Error during network operation: {e.Message}");
                    }

                    isListening = true;
                });
            }
        }
    }


    public async Task SendMessageAsync(string message)
    {
        if (stream != null)
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            await stream.WriteAsync(data, 0, data.Length);
            Debug.Log($"Sent message: {message}");
        }
    }

    private async Task<string> ReceiveMessageAsync()
    {
        string response = null;
        if (stream != null)
        {
            byte[] data = new byte[client.ReceiveBufferSize];
            int bytesRead = await stream.ReadAsync(data, 0, data.Length);
            response = Encoding.UTF8.GetString(data, 0, bytesRead);
            Debug.Log($"Received message: {response}");
        }
        return response;
    }

    private async void OnDestroy()
    {
        await SendMessageAsync("Client disconnected");
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }
    }
}
