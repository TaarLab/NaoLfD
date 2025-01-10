using System.Numerics;
using Mediapipe.Net.Framework.Protobuf;

public static class Extensions
{
    public static Vector3 ToVector3(this NormalizedLandmark landmark) =>
        new(landmark.X, landmark.Y, landmark.Z);
}