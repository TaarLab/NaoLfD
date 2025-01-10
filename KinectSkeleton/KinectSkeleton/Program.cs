using System;
using System.Globalization;
using System.IO;
using System.Linq;
using CoreFramework;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.Kinect;
using OpenCvSharp;

namespace KinectSkeleton;

internal class Program
{
    private static bool _ready;
    private static int _moveCounter;
    private static int _frameCounter;

    public static void Main()
    {
        var sensor =
            KinectSensor.KinectSensors.First(potentialSensor => potentialSensor.Status == KinectStatus.Connected);

        sensor.ColorStream.Enable();
        sensor.DepthStream.Enable();
        sensor.SkeletonStream.Enable();

        var baseFolderPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "NaoPushingData");
        var dateTimeFolder = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        var dateFolderPath = Path.Combine(baseFolderPath, dateTimeFolder);

        if (!Directory.Exists(baseFolderPath))
            Directory.CreateDirectory(baseFolderPath);
        if (!Directory.Exists(dateFolderPath))
            Directory.CreateDirectory(dateFolderPath);
        if (!Directory.Exists(Path.Combine(dateFolderPath, "depth")))
            Directory.CreateDirectory(Path.Combine(dateFolderPath, "depth"));
        if (!Directory.Exists(Path.Combine(dateFolderPath, "color")))
            Directory.CreateDirectory(Path.Combine(dateFolderPath, "color"));

        var dataWriter = new StreamWriter(Path.Combine(dateFolderPath, "data.csv"), append: false);
        var videoWriter = new VideoWriter(Path.Combine(dateFolderPath, "video.mp4"),
            FourCC.MP4V, 30, new Size(640, 480));
        var csvWriter = new CsvWriter(dataWriter,
            new CsvConfiguration(CultureInfo.InvariantCulture) { Delimiter = "," });
        csvWriter.WriteHeader<RobotMovementData>();
        var skeletonProcessor = new SkeletonProcessor();

        sensor.AllFramesReady += (s, e) =>
        {
            using var colorFrame = e.OpenColorImageFrame();
            using var skeletonFrame = e.OpenSkeletonFrame();
            using var depthFrame = e.OpenDepthImageFrame();
            if (colorFrame == null || skeletonFrame == null || depthFrame == null) return;

            var skeletons = new Skeleton[skeletonFrame.SkeletonArrayLength];
            skeletonFrame.CopySkeletonDataTo(skeletons);
            var pixels = new byte[colorFrame.PixelDataLength];
            colorFrame.CopyPixelDataTo(pixels);
            var depthData = new short[depthFrame.PixelDataLength];
            depthFrame.CopyPixelDataTo(depthData);

            var skeleton =
                skeletons.FirstOrDefault(skeleton => skeleton.TrackingState == SkeletonTrackingState.Tracked);
            if (skeleton == null) return;
            var armJoints = new[]
            {
                JointType.ShoulderLeft,
                JointType.ElbowLeft,
                JointType.WristLeft,
                JointType.ShoulderRight,
                JointType.ElbowRight,
                JointType.WristRight
            };

            if (skeleton.Joints.Any(j => armJoints.Contains(j.JointType) && j.TrackingState == JointTrackingState.NotTracked))
            {
                Console.WriteLine("Skip Frame!");
                return;
            }
            var processSkeletonData = skeletonProcessor.ProcessSkeletonData(skeleton);

            var shoulderLeftPos = processSkeletonData.Joints.First(j => j.JointType == JointType.ShoulderLeft).Position;
            var elbowLeftPos = processSkeletonData.Joints.First(j => j.JointType == JointType.ElbowLeft).Position;
            var wristLeftPos = processSkeletonData.Joints.First(j => j.JointType == JointType.WristLeft).Position;
            var shoulderRightPos =
                processSkeletonData.Joints.First(j => j.JointType == JointType.ShoulderRight).Position;
            var elbowRightPos = processSkeletonData.Joints.First(j => j.JointType == JointType.ElbowRight).Position;
            var wristRightPos = processSkeletonData.Joints.First(j => j.JointType == JointType.WristRight).Position;
            
            var shoulderLeftPosColor = sensor.CoordinateMapper.MapSkeletonPointToColorPoint(shoulderLeftPos, ColorImageFormat.RgbResolution640x480Fps30);
            var elbowLeftPosColor = sensor.CoordinateMapper.MapSkeletonPointToColorPoint(elbowLeftPos, ColorImageFormat.RgbResolution640x480Fps30);
            var wristLeftPosColor = sensor.CoordinateMapper.MapSkeletonPointToColorPoint(wristLeftPos, ColorImageFormat.RgbResolution640x480Fps30);
            var shoulderRightPosColor = sensor.CoordinateMapper.MapSkeletonPointToColorPoint(shoulderRightPos, ColorImageFormat.RgbResolution640x480Fps30);
            var elbowRightPosColor = sensor.CoordinateMapper.MapSkeletonPointToColorPoint(elbowRightPos, ColorImageFormat.RgbResolution640x480Fps30);
            var wristRightPosColor = sensor.CoordinateMapper.MapSkeletonPointToColorPoint(wristRightPos, ColorImageFormat.RgbResolution640x480Fps30);

            var shoulderLeft = AngleCalculation.FindShoulder(
                QuaternionExtensions.ToVector3(shoulderLeftPos.X, shoulderLeftPos.Y, shoulderLeftPos.Z),
                QuaternionExtensions.ToVector3(elbowLeftPos.X, elbowLeftPos.Y, elbowLeftPos.Z)
            );
            var shoulderRight = AngleCalculation.FindShoulder(
                QuaternionExtensions.ToVector3(shoulderRightPos.X, shoulderRightPos.Y, shoulderRightPos.Z),
                QuaternionExtensions.ToVector3(elbowRightPos.X, elbowRightPos.Y, elbowRightPos.Z)
            );
            var elbowLeft = AngleCalculation.FindElbow(
                QuaternionExtensions.ToVector3(elbowLeftPos.X, elbowLeftPos.Y, elbowLeftPos.Z),
                QuaternionExtensions.ToVector3(wristLeftPos.X, wristLeftPos.Y, wristLeftPos.Z),
                shoulderLeft,
                Hand.Left
            );
            var elbowRight = AngleCalculation.FindElbow(
                QuaternionExtensions.ToVector3(elbowRightPos.X, elbowRightPos.Y, elbowRightPos.Z),
                QuaternionExtensions.ToVector3(wristRightPos.X, wristRightPos.Y, wristRightPos.Z),
                shoulderRight,
                Hand.Right
            );

            var robotMovementData = new RobotMovementData
            {
                LeftElbowYaw = elbowLeft.Z,
                LeftElbowRoll = elbowLeft.Y,
                RightElbowYaw = elbowRight.Z,
                RightElbowRoll = elbowRight.Y,
                LeftShoulderRoll = shoulderLeft.Y,
                LeftShoulderPitch = shoulderLeft.X,
                RightShoulderRoll = shoulderRight.Y,
                RightShoulderPitch = shoulderRight.X,

                LeftWristX = wristLeftPos.X,
                LeftWristY = wristLeftPos.Y,
                LeftWristZ = wristLeftPos.Z,
                RightWristX = wristRightPos.X,
                RightWristY = wristRightPos.Y,
                RightWristZ = wristRightPos.Z,
                LeftElbowX = elbowLeftPos.X,
                LeftElbowY = elbowLeftPos.Y,
                LeftElbowZ = elbowLeftPos.Z,
                RightElbowX = elbowRightPos.X,
                RightElbowY = elbowRightPos.Y,
                RightElbowZ = elbowRightPos.Z,
                LeftShoulderX = shoulderLeftPos.X,
                LeftShoulderY = shoulderLeftPos.Y,
                LeftShoulderZ = shoulderLeftPos.Z,
                RightShoulderX = shoulderRightPos.X,
                RightShoulderY = shoulderRightPos.Y,
                RightShoulderZ = shoulderRightPos.Z,

                LeftWristColorX = wristLeftPosColor.X,
                LeftWristColorY = wristLeftPosColor.Y,
                RightWristColorY = wristRightPosColor.Y,
                LeftElbowColorX = elbowLeftPosColor.X,
                LeftElbowColorY = elbowLeftPosColor.Y,
                RightElbowColorX = elbowRightPosColor.X,
                RightElbowColorY = elbowRightPosColor.Y,
                LeftShoulderColorX = shoulderLeftPosColor.X,
                LeftShoulderColorY = shoulderLeftPosColor.Y,
                RightShoulderColorX = shoulderRightPosColor.X,
                RightShoulderColorY = shoulderRightPosColor.Y,
            };

            var colorFrameWidth = colorFrame.Width;
            var colorFrameHeight = colorFrame.Height;

            _moveCounter++;
            if (_moveCounter % 5 == 0)
                Robot.MoveRobot(robotMovementData);

            if (_ready)
            {
                _frameCounter++;
                robotMovementData.FrameNumber = _frameCounter;
                robotMovementData.Timestamp = DateTime.Now.ToString("o", CultureInfo.InvariantCulture);
                csvWriter.WriteRecord(robotMovementData);
                csvWriter.NextRecord();
                csvWriter.Flush();


                var mat = new Mat(colorFrameHeight, colorFrameWidth, MatType.CV_8UC4, pixels);
                var matRgb = new Mat();
                Cv2.CvtColor(mat, matRgb, ColorConversionCodes.BGRA2BGR);
//                Cv2.Circle(matRgb, new Point(wristRightPosColor.X, wristRightPosColor.Y), 10, new Scalar(0, 0, 255), -1);
//                Cv2.Circle(matRgb, new Point(wristLeftPosColor.X, wristLeftPosColor.Y), 10, new Scalar(0, 0, 255), -1);
//                Cv2.Circle(matRgb, new Point(elbowRightPosColor.X, elbowRightPosColor.Y), 10, new Scalar(0, 0, 255), -1);
//                Cv2.Circle(matRgb, new Point(elbowLeftPosColor.X, elbowLefrtPosColor.Y), 10, new Scalar(0, 0, 255), -1);
//                Cv2.Circle(matRgb, new Point(shoulderRightPosColor.X, shoulderRightPosColor.Y), 10, new Scalar(0, 0, 255), -1);
//                Cv2.Circle(matRgb, new Point(shoulderLeftPosColor.X, shoulderLeftPosColor.Y), 10, new Scalar(0, 0, 255), -1);
                videoWriter.Write(matRgb);

                var depthFilePath = Path.Combine(dateFolderPath, "depth", $"depth_data_{_frameCounter}.bin");
                using var depthWriter = new BinaryWriter(File.Open(depthFilePath, FileMode.Create));
                foreach (var depthValue in depthData)
                    depthWriter.Write(depthValue);

                var colorFilePath = Path.Combine(dateFolderPath, "color", $"color_data_{_frameCounter}.bin");
                using var colorWriter = new BinaryWriter(File.Open(colorFilePath, FileMode.Create));
                foreach (var colorValue in pixels)
                    colorWriter.Write(colorValue);
            }
            else
            {
                Console.Clear();
                Console.WriteLine(
                    $"Left Elbow Pos: ({elbowLeftPos.X - shoulderLeftPos.X}, {elbowLeftPos.Y - shoulderLeftPos.Y}, {elbowLeftPos.Z - shoulderLeftPos.Z})");
                Console.WriteLine(
                    $"Left Wrist Pos: ({wristLeftPos.X - elbowLeftPos.X}, {wristLeftPos.Y - elbowLeftPos.Y}, {wristLeftPos.Z - elbowLeftPos.Z})");
                Console.WriteLine(
                    $"Right Elbow Pos: ({elbowRightPos.X - shoulderRightPos.X}, {elbowRightPos.Y - shoulderRightPos.Y}, {elbowRightPos.Z - shoulderRightPos.Z})");
                Console.WriteLine(
                    $"Right Wrist Pos: ({wristRightPos.X - elbowRightPos.X}, {wristRightPos.Y - elbowRightPos.Y}, {wristRightPos.Z - elbowRightPos.Z})");

                Console.WriteLine($"Left Shoulder: ({shoulderLeft.X}, {shoulderLeft.Y})");
                Console.WriteLine($"Right Shoulder: ({shoulderRight.X}, {shoulderRight.Y})");
                Console.WriteLine($"Left Elbow: ({elbowLeft.Y}, {elbowLeft.Z})");
                Console.WriteLine($"Right Elbow: ({elbowRight.Y}, {elbowRight.Z})");
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
                break;
            if (key == ConsoleKey.R)
                _ready = true;
        }

        Console.WriteLine("Stopped!");

        sensor.Stop();
        csvWriter.Dispose();
        videoWriter.Release();
    }
}