using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Kinect;
using Newtonsoft.Json;
using OpenCvSharp;

namespace RealWorld;

internal class Program
{
    private static bool _ready;
    private static bool _is_grasp;
    private static dynamic _last_goal;
    private static dynamic _last_origin;
    private static bool _ready_from_api = true;
    private static readonly HttpClient _httpClient = new();

    private static dynamic currentAngles = new
    {
        LElbowYaw = 0,
        LElbowRoll = 0,
        LShoulderRoll = 0,
        LShoulderPitch = 0
    };

    public static async Task Main()
    {
        var sensor =
            KinectSensor.KinectSensors.First(potentialSensor => potentialSensor.Status == KinectStatus.Connected);

        sensor.ColorStream.Enable();
        sensor.DepthStream.Enable();

        sensor.AllFramesReady += async (s, e) =>
        {
            using var colorFrame = e.OpenColorImageFrame();
            using var depthFrame = e.OpenDepthImageFrame();
            if (colorFrame == null || depthFrame == null) return;

            var pixels = new byte[colorFrame.PixelDataLength];
            colorFrame.CopyPixelDataTo(pixels);
            var depthData = new short[depthFrame.PixelDataLength];
            depthFrame.CopyPixelDataTo(depthData);

            var colorFrameWidth = colorFrame.Width;
            var colorFrameHeight = colorFrame.Height;

            if (_ready && _ready_from_api)
            {
                Console.WriteLine("Predict!");
                _ready_from_api = false;
                // _ready = false;
                var mat = new Mat(colorFrameHeight, colorFrameWidth, MatType.CV_8UC4, pixels);
                var matRgb = new Mat();
                Cv2.CvtColor(mat, matRgb, ColorConversionCodes.BGRA2BGR);

                var tempImagePath = Path.Combine(Path.GetTempPath(), "temp_image.jpg");
                matRgb.SaveImage(tempImagePath);

                var originGoal = await PredictObjectsAsync(tempImagePath);
                originGoal = new
                {
                    origin_x = (originGoal.origin_x == null ? _last_origin?.origin_x : originGoal.origin_x) - 10,
                    origin_y = originGoal.origin_y == null ? _last_origin?.origin_y : originGoal.origin_y,
                    goal_x = (originGoal.goal_x == null ? _last_goal?.goal_x : originGoal.goal_x) + 30,
                    goal_y = (originGoal.goal_y == null ? _last_goal?.goal_y : originGoal.goal_y) - 10
                };
                if (originGoal.origin_x != null)
                    _last_origin = new
                    {
                        origin_x = originGoal.origin_x + 10,
                        origin_y = originGoal.origin_y
                    };
                if (originGoal.goal_x != null)
                    _last_goal = new
                    {
                        goal_x = originGoal.goal_x - 30,
                        goal_y = originGoal.goal_y + 10
                    };
                if (originGoal == null || originGoal.origin_x == null || originGoal.goal_x == null)
                {
                    _ready_from_api = true;
                    Console.WriteLine(JsonConvert.SerializeObject(originGoal, Formatting.Indented));
                    return;
                }

                Console.WriteLine(JsonConvert.SerializeObject(originGoal, Formatting.Indented));

                var originZ = GetDepthAtPoint(depthData, colorFrameWidth, colorFrameHeight, (int)originGoal.origin_x,
                    (int)originGoal.origin_y);
                var goalZ = GetDepthAtPoint(depthData, colorFrameWidth, colorFrameHeight, (int)originGoal.goal_x,
                    (int)originGoal.goal_y);

                if (currentAngles.LShoulderRoll == 0 && currentAngles.LShoulderPitch == 0 &&
                    currentAngles.LElbowRoll == 0)
                    currentAngles = await GetRobotAnglesAsync();

                var data = new
                {
                    LeftElbowYaw = currentAngles.LElbowYaw,
                    LeftElbowRoll = currentAngles.LElbowRoll,
                    LeftShoulderRoll = currentAngles.LShoulderRoll,
                    LeftShoulderPitch = currentAngles.LShoulderPitch,
                    origin_x = originGoal.origin_x - 10,
                    origin_y = originGoal.origin_y,
                    origin_z = originZ,
                    goal_x = originGoal.goal_x + 30,
                    goal_y = originGoal.goal_y - 10,
                    goal_z = goalZ,
                    is_grasped = _is_grasp ? 1 : -1
                };
                Console.WriteLine(JsonConvert.SerializeObject(data, Formatting.Indented));

                var jointPrediction = await PredictJointsAsync(data);

                Console.WriteLine(JsonConvert.SerializeObject(jointPrediction, Formatting.Indented));

                var moveArmsData = new
                {
                    left_shoulder_pitch = (double)jointPrediction.LeftShoulderPitch,
                    left_shoulder_roll = (double)jointPrediction.LeftShoulderRoll,
                    left_elbow_yaw = (double)jointPrediction.LeftElbowYaw,
                    left_elbow_roll = (double)jointPrediction.LeftElbowRoll,
                    // left_shoulder_pitch = 45,
                    // left_shoulder_roll = 45,
                    // left_elbow_yaw = 0,
                    // left_elbow_roll = 0,
                    right_shoulder_pitch = 0.0,
                    right_shoulder_roll = 0.0,
                    right_elbow_yaw = 0.0,
                    right_elbow_roll = 0.0
                };

                await MoveArmsAsync(moveArmsData);
                currentAngles.LShoulderPitch = jointPrediction.LeftShoulderPitch;
                currentAngles.LShoulderRoll = jointPrediction.LeftShoulderRoll;
                currentAngles.LElbowRoll = jointPrediction.LeftElbowRoll;
                currentAngles.LElbowYaw = jointPrediction.LeftElbowYaw;
                _ready_from_api = true;
            }
        };

        try
        {
            sensor.Start();
        }
        catch (IOException)
        {
        }

        while (true)
        {
            var key = Console.ReadKey().Key;
            if (key == ConsoleKey.Q)
            {
                sensor.Stop();
                break;
            }

            if (key == ConsoleKey.R)
                _ready = true;
            if (key == ConsoleKey.G)
                _is_grasp = true;
        }

        Console.WriteLine("Stopped!");

        sensor.Stop();
    }

    private static double GetDepthAtPoint(short[] depthData, int width, int height, int x, int y)
    {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        var index = (y * width) + x;
        return depthData[index] / 10000.0;
    }

    private static async Task<dynamic> PredictObjectsAsync(string imagePath)
    {
        using var content = new MultipartFormDataContent();
        var imageContent = new ByteArrayContent(File.ReadAllBytes(imagePath));
        imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
        content.Add(imageContent, "image", Path.GetFileName(imagePath));

        var response = await _httpClient.PostAsync("http://127.0.0.1:5000/predict_objects", content);
        if (!response.IsSuccessStatusCode) return null;

        var jsonResponse = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<dynamic>(jsonResponse);
    }

    private static async Task<dynamic> PredictJointsAsync(dynamic fakeJointData)
    {
        var jsonContent = JsonConvert.SerializeObject(fakeJointData);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("http://127.0.0.1:5000/predict_joints", content);
        if (!response.IsSuccessStatusCode) return null;

        var jsonResponse = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<dynamic>(jsonResponse);
    }

    private static async Task<dynamic> GetRobotAnglesAsync()
    {
        var response = await _httpClient.GetAsync("http://127.0.0.1:5001/get_angles");
        if (!response.IsSuccessStatusCode) return null;

        var jsonResponse = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<dynamic>(jsonResponse);
    }

    private static async Task MoveArmsAsync(dynamic moveArmsData)
    {
        var jsonContent = JsonConvert.SerializeObject(moveArmsData);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("http://127.0.0.1:5001/move_arms", content);
        if (!response.IsSuccessStatusCode)
        {
            Console.WriteLine("Failed to move arms.");
        }
    }
}