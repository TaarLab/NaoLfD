using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace CoreFramework;

public static class Robot
{
    public static async Task MoveRobot(RobotMovementData robotMovementData)
    {
        try
        {
            var payload = new
            {
                yaw = 0,
                pitch = 0,
                emotion = "",
                left_eye_is_closed = false,
                right_eye_is_closed = false,
                Left_Elbow_Yaw = robotMovementData.LeftElbowYaw,
                Left_Elbow_Roll = robotMovementData.LeftElbowRoll,
                Right_Elbow_Yaw = robotMovementData.RightElbowYaw,
                Right_Elbow_Roll = robotMovementData.RightElbowRoll,
                Left_Shoulder_Pitch = robotMovementData.LeftShoulderPitch,
                Left_Shoulder_Roll = robotMovementData.LeftShoulderRoll - 17,
                Right_Shoulder_Pitch = robotMovementData.RightShoulderPitch,
                Right_Shoulder_Roll = robotMovementData.RightShoulderRoll + 17,
            };

            var json = JsonConvert.SerializeObject(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await new HttpClient().PostAsync("http://127.0.0.1:5001/move_arms", content);
            response.EnsureSuccessStatusCode();
        }
        catch (Exception e)
        {
            Console.WriteLine("Exception: " + e.Message);
        }
    }
}