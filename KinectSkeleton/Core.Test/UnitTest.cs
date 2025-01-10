using Xunit;
using System.Numerics;
using Xunit.Abstractions;

namespace Core.Test;

public class UnitTest(ITestOutputHelper testOutputHelper)
{
    [Fact]
    public void ElbowTest()
    {
        Vector3 elbow;

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0, 0, -1)); // Front

        Assert.InRange(elbow.Y, -0.01, 0.01);
        Assert.InRange(elbow.Z, -0.01, 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(1, 0, 0)); // Right

        Assert.InRange(elbow.Y, -0.01, 0.01);
        Assert.InRange(elbow.Z, -90 - 0.01, -90 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(-1, 0, 0)); // Left

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, -90 - 0.01, -90 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(-0.5f, 0, -0.5f)); // Left, Front

        Assert.InRange(elbow.Y, 45 - 0.01, 45 + 0.01);
        Assert.InRange(elbow.Z, -0.01, 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(-0.5f, 0, 0.5f)); // Left, Back

        Assert.InRange(elbow.Y, 135 - 0.01, 135 + 0.01);
        Assert.InRange(elbow.Z, -0.01, 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0, 1, 0)); // Up

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, 90 - 0.01, 90 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0, 0.5f, -0.5f)); // Up, Front

        Assert.InRange(elbow.Y, 45 - 0.01, 45 + 0.01);
        Assert.InRange(elbow.Z, 90 - 0.01, 90 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0, 0.5f, 0.5f)); // Up, Back

        Assert.InRange(elbow.Y, 135 - 0.01, 135 + 0.01);
        Assert.InRange(elbow.Z, 90 - 0.01, 90 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0, -1, 0)); // Down

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, -90 - 0.01, -90 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0.5f, 0.5f, 0)); // Right, Up

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, 135 - 0.01, 135 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(0.5f, -0.5f, 0)); // Right, Down

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, -135 - 0.01, -135 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(-0.5f, 0.5f, 0)); // Left, Up

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, 45 - 0.01, 45 + 0.01);

        elbow = AngleCalculation.FindElbow(new Vector3(0, 0, 0), new Vector3(-0.5f, -0.5f, 0)); // Left, Down

        Assert.InRange(elbow.Y, 90 - 0.01, 90 + 0.01);
        Assert.InRange(elbow.Z, -45 - 0.01, -45 + 0.01);
    }

    private void ApplyOnRobot(Vector3 rightShoulderVector, Vector3 leftShoulderVector = new(),
        Vector3 rightElbowVector = new(), Vector3 leftElbowVector = new(), int delay = 2000)
    {
        // Thread.Sleep(delay);
        // Robot.MoveRobot(0, 0, false, false, "",
        //     leftShoulderVector, leftElbowVector,
        //     rightShoulderVector, rightElbowVector
        // );
    }

    [Fact]
    public async void MoveTest()
    {
        await Robot.MoveRobot(0, 0, false, false, "",
            new Vector3(), new Vector3(),
            new Vector3(), new Vector3()
        );
    }

    [Fact]
    public void ShoulderTest()
    {
        Vector3 shoulder;

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), new Vector3(0, 0, -1)); // Front
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, -0.01, 0.01);
        Assert.InRange(shoulder.Y, -0.01, 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), new Vector3(0, -1, 0)); // Down
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, 90 - 0.01, 90 + 0.01);
        Assert.InRange(shoulder.Y, -0.01, 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), new Vector3(0, 1, 0)); // Up
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, -90 - 0.01, -90 + 0.01);
        Assert.InRange(shoulder.Y, -0.01, 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0),
            Vector3.Normalize(new Vector3(0, -0.5f, -0.5f))); // Down, Front
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, 45 - 0.01, 45 + 0.01);
        Assert.InRange(shoulder.Y, -0.01, 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0),
            Vector3.Normalize(new Vector3(0.5f, -0.5f, 0))); // Down, Right
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, 90 - 0.01, 90 + 0.01);
        Assert.InRange(shoulder.Y, -45 - 0.01, -45 + 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0),
            Vector3.Normalize(new Vector3(0.5f, 0.5f, 0))); // Up, Right
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, -90 - 0.01, -90 + 0.01);
        Assert.InRange(shoulder.Y, -45 - 0.01, -45 + 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), new Vector3(1, 0, 0)); // Right
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, -0.01, 0.01);
        Assert.InRange(shoulder.Y, -90 - 0.01, -90 + 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0),
            Vector3.Normalize(new Vector3(0.5f, 0, -0.5f))); // Front, Right
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, -0.01, 0.01);
        Assert.InRange(shoulder.Y, -45 - 0.01, -45 + 0.01);

        shoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0),
            Vector3.Normalize(new Vector3(0.5f, 0.5f, -0.5f))); // Front, Up, Right
        ApplyOnRobot(shoulder);

        Assert.InRange(shoulder.X, -45 - 0.01, -45 + 0.01);
        Assert.InRange(shoulder.Y, -35 - 0.01, -35 + 0.01);
    }

    [Fact]
    public void MoveShoulderTest()
    {
        const float steps = 50f;
        // Up to Down, Right/Left
        for (var i = 0; i <= steps; i++)
        {
            var angle = (float)Math.PI * i / steps;

            var rightShoulderPosition = new Vector3((float)Math.Sin(angle), (float)Math.Cos(angle), 0);
            var rightShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), rightShoulderPosition);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightShoulderPosition} -> {rightShoulder}");

            var leftShoulderPosition = new Vector3(-(float)Math.Sin(angle), (float)Math.Cos(angle), 0);
            var leftShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), leftShoulderPosition);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftShoulderPosition} -> {leftShoulder}");

            ApplyOnRobot(rightShoulder, leftShoulderVector: leftShoulder, delay: 100);

            if (i == 0)
                Thread.Sleep(2000);
        }

        // Up to Down, Front
        for (var i = 0; i <= steps; i++)
        {
            var angle = (float)Math.PI * i / steps;

            var rightShoulderPosition = new Vector3(0, (float)Math.Cos(angle), -(float)Math.Sin(angle));
            var rightShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), rightShoulderPosition);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightShoulderPosition} -> {rightShoulder}");

            var leftShoulderPosition = new Vector3(0, (float)Math.Cos(angle), -(float)Math.Sin(angle));
            var leftShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), leftShoulderPosition);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftShoulderPosition} -> {leftShoulder}");

            ApplyOnRobot(rightShoulder, leftShoulderVector: leftShoulder, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Up to Down, Right/Left, Front
        for (var i = 0; i <= steps; i++)
        {
            var angle = (float)Math.PI * i / steps;

            var rightShoulderPosition =
                new Vector3((float)Math.Sin(angle), (float)Math.Cos(angle), -(float)Math.Sin(angle));
            var rightShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), rightShoulderPosition);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightShoulderPosition} -> {rightShoulder}");

            var leftShoulderPosition =
                new Vector3(-(float)Math.Sin(angle), (float)Math.Cos(angle), -(float)Math.Sin(angle));
            var leftShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), leftShoulderPosition);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftShoulderPosition} -> {leftShoulder}");

            ApplyOnRobot(rightShoulder, leftShoulderVector: leftShoulder, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Front to Right/Left
        for (var i = 0; i <= steps; i++)
        {
            var angle = (float)Math.PI * i / (steps * 2);

            var rightShoulderPosition = new Vector3((float)Math.Sin(angle), 0, -(float)Math.Cos(angle));
            var rightShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), rightShoulderPosition);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightShoulderPosition} -> {rightShoulder}");

            var leftShoulderPosition = new Vector3(-(float)Math.Sin(angle), 0, -(float)Math.Cos(angle));
            var leftShoulder = AngleCalculation.FindShoulder(new Vector3(0, 0, 0), leftShoulderPosition);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftShoulderPosition} -> {leftShoulder}");

            ApplyOnRobot(rightShoulder, leftShoulderVector: leftShoulder, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }
    }

    [Fact]
    public void MoveElbowTest()
    {
        const float steps = 50f;
        // Right Elbow, Up Right to Down Right
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (-120 + 240 * i / steps);

            var rightElbowPosition = new Vector3(-(float)Math.Cos(angle), -(float)Math.Sin(angle), 0);
            var rightElbow = AngleCalculation.FindElbow(new Vector3(), rightElbowPosition, hand: Hand.Right);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightElbowPosition} -> {rightElbow}");

            ApplyOnRobot(new Vector3(), leftShoulderVector: new Vector3(90, 0, 0), rightElbowVector: rightElbow,
                delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Left Elbow, Up Right to Down Right
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (-120 + 240 * i / steps);

            var leftElbowPosition = new Vector3((float)Math.Cos(angle), -(float)Math.Sin(angle), 0);
            var leftElbow = AngleCalculation.FindElbow(new Vector3(), leftElbowPosition, hand: Hand.Left);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftElbowPosition} -> {leftElbow}");

            ApplyOnRobot(new Vector3(90, 0, 0), leftElbowVector: leftElbow, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Up to Down
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (-90 + 180 * i / steps);

            var rightElbowPosition = new Vector3(0, -(float)Math.Sin(angle), -(float)Math.Cos(angle));
            var rightElbow = AngleCalculation.FindElbow(new Vector3(), rightElbowPosition, hand: Hand.Right);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightElbowPosition} -> {rightElbow}");

            var leftElbowPosition = new Vector3(0, -(float)Math.Sin(angle), -(float)Math.Cos(angle));
            var leftElbow = AngleCalculation.FindElbow(new Vector3(), leftElbowPosition, hand: Hand.Left);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftElbowPosition} -> {leftElbow}");

            ApplyOnRobot(new Vector3(), rightElbowVector: rightElbow, leftElbowVector: leftElbow, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Right Elbow, Left to Right
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (-90 + 180 * i / steps);

            var rightElbowPosition = new Vector3((float)Math.Sin(angle), 0, -(float)Math.Cos(angle));
            var rightElbow = AngleCalculation.FindElbow(new Vector3(), rightElbowPosition, hand: Hand.Right);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightElbowPosition} -> {rightElbow}");

            ApplyOnRobot(new Vector3(), leftShoulderVector: new Vector3(90, 0, 0), rightElbowVector: rightElbow,
                delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Left Elbow, Left to Right
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (-90 + 180 * i / steps);

            var leftElbowPosition = new Vector3(-(float)Math.Sin(angle), 0, -(float)Math.Cos(angle));
            var leftElbow = AngleCalculation.FindElbow(new Vector3(), leftElbowPosition, hand: Hand.Left);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftElbowPosition} -> {leftElbow}");

            ApplyOnRobot(new Vector3(90, 0, 0), rightElbowVector: new Vector3(), leftElbowVector: leftElbow,
                delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Shoulder Down, Front to Back
        var rightShoulderRotation = new Vector3(90, 0, 0);
        var leftShoulderRotation = new Vector3(90, 0, 0);
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (-90 + 180 * i / steps) + (90 / 180.0 * Math.PI);

            var rightElbowPosition = new Vector3(0, -(float)Math.Sin(angle), -(float)Math.Cos(angle));
            var rightElbow = AngleCalculation.FindElbow(new Vector3(), rightElbowPosition, rightShoulderRotation,
                hand: Hand.Right);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightElbowPosition} -> {rightElbow}");

            var leftElbowPosition = new Vector3(0, -(float)Math.Sin(angle), -(float)Math.Cos(angle));
            var leftElbow = AngleCalculation.FindElbow(new Vector3(), leftElbowPosition, leftShoulderRotation,
                hand: Hand.Left);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftElbowPosition} -> {leftElbow}");

            ApplyOnRobot(rightShoulderRotation, leftShoulderVector: leftShoulderRotation, rightElbowVector: rightElbow,
                leftElbowVector: leftElbow, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }

        // Shoulder Right/Left, Front to Back
        rightShoulderRotation = new Vector3(0, -90, 0);
        leftShoulderRotation = new Vector3(0, 90, 0);
        for (var i = 0; i <= steps; i++)
        {
            var angle = Math.PI / 180 * (120 * i / steps);

            var rightElbowPosition = new Vector3((float)Math.Sin(angle), 0, -(float)Math.Cos(angle));
            var rightElbow = AngleCalculation.FindElbow(new Vector3(), rightElbowPosition, rightShoulderRotation,
                hand: Hand.Right);
            testOutputHelper.WriteLine($"Right {angle / Math.PI * 180}: {rightElbowPosition} -> {rightElbow}");

            var leftElbowPosition = new Vector3(-(float)Math.Sin(angle), 0, -(float)Math.Cos(angle));
            var leftElbow = AngleCalculation.FindElbow(new Vector3(), leftElbowPosition, leftShoulderRotation,
                hand: Hand.Left);
            testOutputHelper.WriteLine($"Left {angle / Math.PI * 180}: {leftElbowPosition} -> {leftElbow}");

            ApplyOnRobot(rightShoulderRotation, leftShoulderVector: leftShoulderRotation, rightElbowVector: rightElbow,
                leftElbowVector: leftElbow, delay: 100);
            if (i == 0)
                Thread.Sleep(2000);
        }
    }
}