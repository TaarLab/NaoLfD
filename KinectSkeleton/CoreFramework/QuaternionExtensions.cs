using System;
using System.Numerics;

namespace CoreFramework;

public static class QuaternionExtensions
{
    private const float Epsilon = 1e-7f;
    private const float Pi = (float)Math.PI;
    
    public static Vector3 ToVector3(float x, float y, float z) => new(x, y, z);

    public static Vector3 ToEuler(this Quaternion quat, string seq = "xyz")
    {
        seq = seq.ToLower();
        var i = ElementaryBasisIndex(seq[0]);
        var j = ElementaryBasisIndex(seq[1]);
        var k = ElementaryBasisIndex(seq[2]);

        var symmetric = i == k;
        if (symmetric)
        {
            k = 3 - i - j;
        }

        var sign = (i - j) * (j - k) * (k - i) / 2;

        double a, b, c, d;
        if (symmetric)
        {
            a = quat.W;
            b = GetElement(quat, i);
            c = GetElement(quat, j);
            d = GetElement(quat, k) * sign;
        }
        else
        {
            a = quat.W - GetElement(quat, j);
            b = GetElement(quat, i) + GetElement(quat, k) * sign;
            c = GetElement(quat, j) + quat.W;
            d = GetElement(quat, k) * sign - GetElement(quat, i);
        }

        return GetAngles(symmetric, sign, Pi / 2, a, b, c, d);
    }

    private static float GetElement(Quaternion q, int index)
    {
        return index switch
        {
            0 => q.X,
            1 => q.Y,
            2 => q.Z,
            _ => throw new ArgumentException("Invalid index")
        };
    }

    private static float GetElement(Vector3 q, int index)
    {
        return index switch
        {
            0 => q.X,
            1 => q.Y,
            2 => q.Z,
            _ => throw new ArgumentException("Invalid index")
        };
    }

    private static void SetElement(ref Vector3 q, int index, float value)
    {
        switch (index)
        {
            case 0:
            {
                q.X = value;
                break;
            }
            case 1:
            {
                q.Y = value;
                break;
            }
            case 2:
            {
                q.Z = value;
                break;
            }
            default: throw new ArgumentException("Invalid index");
        }
    }

    private static int ElementaryBasisIndex(char axis)
    {
        return axis switch
        {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            _ => throw new ArgumentException("Invalid axis")
        };
    }

    private static Vector3 GetAngles(bool symmetric, int sign, double lambda, double a, double b, double c,
        double d)
    {
        var angles = new Vector3
        {
            Y = (float)(2 * Math.Atan2(Math.Sqrt(c * c + d * d), Math.Sqrt(a * a + b * b)))
        };

        int caseNum;
        if (Math.Abs(angles.Y) <= Epsilon)
            caseNum = 1;
        else if (Math.Abs(angles.Y - Pi) <= Epsilon)
            caseNum = 2;
        else
            caseNum = 0; // normal case

        var halfSum = Math.Atan2(b, a);
        var halfDiff = Math.Atan2(d, c);

        if (caseNum == 0) // no singularities
        {
            angles.X = (float)(halfSum - halfDiff);
            angles.Z = (float)(halfSum + halfDiff);
        }
        else // any degenerate case
        {
            angles.Z = 0;
            if (caseNum == 1)
                angles.X = (float)(2f * halfSum);
            else
                angles.X = (float)(-2f * halfDiff);
        }

        if (!symmetric)
        {
            angles.Z *= sign;
            angles.Y -= (float)lambda;
        }

        NormalizeAngles(ref angles);

        if (caseNum != 0)
            Console.WriteLine("Gimbal lock detected. Setting third angle to zero.");


        angles.Z *= (float)(180.0f / Math.PI);
        angles.Y *= (float)(180.0f / Math.PI);
        angles.X *= (float)(180.0f / Math.PI);
        return angles;
    }

    private static void NormalizeAngles(ref Vector3 angles)
    {
        for (var idx = 0; idx < 3; idx++)
            if (GetElement(angles, idx) < -Pi)
                SetElement(ref angles, idx, GetElement(angles, idx) + 2f * Pi);
            else if (GetElement(angles, idx) > Pi)
                SetElement(ref angles, idx, GetElement(angles, idx) - 2f * Pi);
    }
}