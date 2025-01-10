using System.Numerics;
using System.Text;
using Newtonsoft.Json;

namespace Core;

public static class Robot
{
    public static async Task MoveRobot(double yawAngle, double pitchAngle, bool left, bool right, string emotion, 
        Vector3 leftShoulder, Vector3 leftElbow, Vector3 rightShoulder, Vector3 rightElbow)
    {
        try
        {
            var payload = new
            {
                yaw = yawAngle,
                pitch = pitchAngle,
                left_eye_is_closed = left,
                right_eye_is_closed = right,
                emotion,
                Left_Shoulder_Pitch = leftShoulder.X,
                Left_Shoulder_Roll = leftShoulder.Y,
                Left_Elbow_Roll = leftElbow.Y,
                Left_Elbow_Yaw = leftElbow.Z,
                Right_Shoulder_Pitch = rightShoulder.X,
                Right_Shoulder_Roll = rightShoulder.Y,
                Right_Elbow_Roll = rightElbow.Y,
                Right_Elbow_Yaw = rightElbow.Z
            };

            var json = JsonConvert.SerializeObject(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await new HttpClient().PostAsync("http://127.0.0.1:5188", content);
            response.EnsureSuccessStatusCode();
        }
        catch (Exception e)
        {
            Console.WriteLine("Exception: " + e.Message);
        }
    }
}