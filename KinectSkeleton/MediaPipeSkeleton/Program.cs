using Core;
using OpenCvSharp;
using Mediapipe.Net.Framework;
using Google.Protobuf.Collections;
using Mediapipe.Net.Framework.Port;
using Mediapipe.Net.Framework.Format;
using Mediapipe.Net.Framework.Packets;
using Mediapipe.Net.Framework.Protobuf;

long simulatedTimestamp = 0;
RepeatedField<NormalizedLandmark>? landmark = null;

var graph = new CalculatorGraph(File.ReadAllText("pose_landmark_cpu.pbtxt"));
graph.ObserveOutputStream("pose_landmarks", 0, (_, _, packetPtr) =>
{
    try
    {
        var packet = Packet<NormalizedLandmarkList>.Create<NormalizedLandmarkListPacket>(packetPtr, false);
        landmark = packet?.Get().Landmark;

        if (landmark != null)
        {
            var noisePos = landmark[0].ToVector3();
            var leftEyePos = landmark[2].ToVector3();
            var rightEyePos = landmark[5].ToVector3();

            var headPitch = AngleCalculation.FindHeadPitch(leftEyePos, rightEyePos, noisePos);
            var headYaw = AngleCalculation.FindHeadYaw(leftEyePos, rightEyePos);

            Console.WriteLine($"Head: ({headPitch}, {headYaw})");

            // Robot.MoveRobot(headYaw, headPitch, false, false, "",
            //     new Vector3(), new Vector3(), new Vector3(), new Vector3());
        }

        return Status.StatusArgs.Ok();
    }
    catch (Exception ex)
    {
        return Status.StatusArgs.Internal(ex.ToString());
    }
});
SidePacket sidePackets = new();
sidePackets.Emplace("use_prev_landmarks", new BoolPacket(false));
sidePackets.Emplace("model_complexity", new IntPacket(1));
sidePackets.Emplace("smooth_landmarks", new BoolPacket(false));
sidePackets.Emplace("enable_segmentation", new BoolPacket(false));
sidePackets.Emplace("smooth_segmentation", new BoolPacket(false));
graph.StartRun(sidePackets).AssertOk();

using var videoCapture = new VideoCapture(0);
videoCapture.Set(VideoCaptureProperties.FrameWidth, 640);
videoCapture.Set(VideoCaptureProperties.FrameHeight, 480);

while (true)
{
    using var frame = new Mat();
    if (!videoCapture.Read(frame)) break;

    var imageFrame = ConvertMatToImageFrame(frame);

    graph.AddPacketToInputStream("image", new ImageFramePacket(imageFrame, new Timestamp(simulatedTimestamp)));
    simulatedTimestamp += 1;

    graph.WaitUntilIdle();

    if (landmark != null)
        DrawPoseLandmarks(frame, landmark);

    if (Cv2.WaitKey(100) == 'q')
        break;
}

return;


static ImageFrame ConvertMatToImageFrame(Mat mat)
{
    return new ImageFrame(ImageFormat.Types.Format.Srgb, mat.Width, mat.Height, (int)mat.Step(), mat.Data, _ => { });
}

static void DrawPoseLandmarks(Mat frame, RepeatedField<NormalizedLandmark> landmarks)
{
    foreach (var landmark in landmarks)
    {
        var x = (int)(landmark.X * frame.Width);
        var y = (int)(landmark.Y * frame.Height);
        Cv2.Circle(frame, new Point(x, y), 5, Scalar.Red, -1);
    }

    Cv2.ImShow("Pose Detection", frame);
}