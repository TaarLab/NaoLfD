using System;
using System.Numerics;

namespace CoreFramework;

public enum Hand
{
    Right,
    Left
}

public static class AngleCalculation
{
    public static Vector3 FindShoulder(Vector3 shoulder, Vector3 elbow)
    {
        var vecBaselineX = new Vector3(1.0f, 0.0f, 0.0f);
        var vecBaselineZ = new Vector3(0.0f, 0.0f, 1.0f);

        var roll = CalculateAngleWithRotation(vecBaselineX, shoulder, elbow, 'y', "XYZ").Z;

        var pitchVector = CalculateAngleWithRotation(vecBaselineZ, shoulder, elbow, 'x', "XYZ", roll);
        var c = (Math.Abs(Math.Abs(pitchVector.X)) < 0.01 && Math.Abs(Math.Abs(pitchVector.Z)) < 0.01) ||
                pitchVector is { X: 90, Y: 0, Z: 0 };
        var pitch = c ? pitchVector.Y : CalculateAngleWithRotation(vecBaselineZ, shoulder, elbow, 'x', "YXZ", roll).X;

        if (float.IsNaN(pitch))
            pitch = 0;
        if (Math.Abs(Math.Abs(pitch) - 180) < 0.01)
            pitch = 0;

        return new Vector3((float)Math.Round(roll), (float)Math.Round(pitch), 0);
    }

    public static Vector3 FindElbow(Vector3 elbow, Vector3 wrist, Vector3 shoulderRotation = new(),
        Hand hand = Hand.Right)
    {
        shoulderRotation = shoulderRotation / 180f * (float)Math.PI;
        var shoulderRotationQ =
            Quaternion.CreateFromYawPitchRoll(-shoulderRotation.Y, shoulderRotation.X, -shoulderRotation.Z);
        elbow = Vector3.Transform(elbow, shoulderRotationQ);
        wrist = Vector3.Transform(wrist, shoulderRotationQ);

        var vecBaselineX = new Vector3(1.0f, 0.0f, 0.0f);

        if (hand == Hand.Left)
            vecBaselineX.X = -1.0f;

        var vecBaselineZ = new Vector3(0.0f, 0.0f, 1.0f);

        var yaw = CalculateAngleWithRotation(vecBaselineX, elbow, wrist, 'z', "ZYX", 0).X;
        yaw *= -1;

        var pitchVector = CalculateAngleWithRotation(vecBaselineZ, elbow, wrist, 'z', "ZYX", yaw);
        var c = (Math.Abs(Math.Abs(pitchVector.X)) < 0.01 && Math.Abs(Math.Abs(pitchVector.Z)) < 0.01) ||
                pitchVector is { X: 90, Y: 0, Z: 0 };
        var pitch = c ? pitchVector.Y : CalculateAngleWithRotation(vecBaselineZ, elbow, wrist, 'z', "YXZ", yaw).X;

        if (float.IsNaN(pitch))
            pitch = 0;
        if (Math.Abs(Math.Abs(pitch) - 180) < 0.01)
            pitch = 0;

        return new Vector3(0, (float)Math.Round(pitch), (float)Math.Round(yaw));
    }
    
    private static Vector3 CalculateAngleWithRotation(Vector3 vecBaseline, Vector3 vector1,
        Vector3 vector2, char axisRotation, string seq, float rotation = 90.0f)
    {
        var vecNew = Vector3.Subtract(vector1, vector2);
        vecNew = Vector3.Normalize(vecNew);
        vecNew = RotateVector(vecNew, axisRotation, rotation);

        return CalculateAngle(vecBaseline, vecNew, seq);
    }

    private static Vector3 CalculateAngle(Vector3 vecBaseline, Vector3 vector, string seq)
    {
        var crossProd = Vector3.Cross(vecBaseline, vector);
        var dotProd = Vector3.Dot(vecBaseline, vector);
        if (crossProd is { Z: 0, Y: 0, X: 0 })
            return new Vector3(90, 0, 0);

        var axis = Vector3.Normalize(crossProd);
        var angle = Math.Acos(dotProd);

        return Quaternion.CreateFromAxisAngle(axis, (float)angle).ToEuler(seq);
    }

    private static Vector3 RotateVector(Vector3 vector, char axis, float degrees)
    {
        var rotation = axis switch
        {
            'x' => Quaternion.CreateFromAxisAngle(Vector3.UnitX, degrees * ((float)Math.PI / 180f)),
            'y' => Quaternion.CreateFromAxisAngle(Vector3.UnitY, degrees * ((float)Math.PI / 180f)),
            'z' => Quaternion.CreateFromAxisAngle(Vector3.UnitZ, degrees * ((float)Math.PI / 180f)),
            _ => throw new ArgumentException("Invalid rotation axis")
        };
        return Vector3.Transform(vector, rotation);
    }
}