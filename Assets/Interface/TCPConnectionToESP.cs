
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class TCPConnectionToESP : MonoBehaviour
{
    private const string serverIP = "192.168.4.1";
    private const int serverPort = 80;

    private TcpClient client;
    private NetworkStream stream;

    private async void Start()
    {
        await ConnectToServerAsync();
    }

    private async Task ConnectToServerAsync()
    {
        try
        {
            client = new TcpClient();
            await client.ConnectAsync(serverIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to server");

        }
        catch (Exception e)
        {
            Debug.Log($"Error connecting to server: {e.Message}");
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
