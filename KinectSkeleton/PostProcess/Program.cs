using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using CoreFramework;
using CsvHelper;
using CsvHelper.Configuration;

public class Program
{
    public static void Main(string[] args)
    {
        var baseFolder = Environment.SpecialFolder.Desktop + "\\NaoRobotProjectData";

        foreach (var folder in Directory.GetDirectories(baseFolder))
        {
            var filePath = Path.Combine(folder, "processed_step_1.csv");
            if (File.Exists(filePath)) 
                ProcessCsvFile(filePath);
        }
    }

    private static void ProcessCsvFile(string filePath)
    {
        var records = new List<RobotMovementData>();

        using (var reader = new StreamReader(filePath))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            csv.Read();
            csv.ReadHeader();
            while (csv.Read())
            {
                var shoulderLeftPos = new Vector3(csv.GetField<float>("LeftShoulderX"), csv.GetField<float>("LeftShoulderY"), csv.GetField<float>("LeftShoulderZ"));
                var elbowLeftPos = new Vector3(csv.GetField<float>("LeftElbowX"), csv.GetField<float>("LeftElbowY"), csv.GetField<float>("LeftElbowZ"));
                var wristLeftPos = new Vector3(csv.GetField<float>("LeftWristX"), csv.GetField<float>("LeftWristY"), csv.GetField<float>("LeftWristZ"));
                
                var shoulderRightPos = new Vector3(csv.GetField<float>("RightShoulderX"), csv.GetField<float>("RightShoulderY"), csv.GetField<float>("RightShoulderZ"));
                var elbowRightPos = new Vector3(csv.GetField<float>("RightElbowX"), csv.GetField<float>("RightElbowY"), csv.GetField<float>("RightElbowZ"));
                var wristRightPos = new Vector3(csv.GetField<float>("RightWristX"), csv.GetField<float>("RightWristY"), csv.GetField<float>("RightWristZ"));

                // Calculate angles
                var shoulderLeft = AngleCalculation.FindShoulder(shoulderLeftPos, elbowLeftPos);
                var shoulderRight = AngleCalculation.FindShoulder(shoulderRightPos, elbowRightPos);
                var elbowLeft = AngleCalculation.FindElbow(elbowLeftPos, wristLeftPos, shoulderLeft, Hand.Left);
                var elbowRight = AngleCalculation.FindElbow(elbowRightPos, wristRightPos, shoulderRight);

                // Create RobotMovementData object
                var data = new RobotMovementData
                {
                    FrameNumber = csv.GetField<int>("FrameNumber"),
                    Timestamp = csv.GetField<string>("Timestamp"),
                    
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
                };

                records.Add(data);
            }
        }

        // Write the new data to a CSV file
        var outputFilePath = Path.Combine(Path.GetDirectoryName(filePath), "processed_step_2.csv");
        using (var writer = new StreamWriter(outputFilePath))
        using (var csv = new CsvWriter(writer, new CsvConfiguration(CultureInfo.InvariantCulture) { Delimiter = "," }))
            csv.WriteRecords(records);
    }
}