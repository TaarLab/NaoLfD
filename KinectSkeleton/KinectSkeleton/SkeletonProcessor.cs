using System;
using System.Collections.Generic;
using Microsoft.Kinect;

namespace KinectSkeleton;
public class FilteredSkeleton
{
    public int TrackingId { get; set; }
    public SkeletonTrackingState TrackingState { get; set; }
    public List<FilteredJoint> Joints { get; set; } = [];
}

public class FilteredJoint
{
    public JointType JointType { get; set; }
    public SkeletonPoint Position { get; set; }
}

public class SkeletonProcessor
{
    private readonly Dictionary<JointType, KalmanFilter[]> _kalmanFilters;

    public SkeletonProcessor()
    {
        _kalmanFilters = new Dictionary<JointType, KalmanFilter[]>();

        foreach (JointType jointType in Enum.GetValues(typeof(JointType)))
        {
            _kalmanFilters[jointType] = new KalmanFilter[3];
            for (var i = 0; i < 3; i++) 
                _kalmanFilters[jointType][i] = new KalmanFilter(processNoise: 0.1, measurementNoise: 1, initialValue: 0.0);
        }
    }

    public FilteredSkeleton ProcessSkeletonData(Skeleton skeleton)
    {
        var filteredSkeleton = new FilteredSkeleton
        {
            TrackingId = skeleton.TrackingId,
            TrackingState = skeleton.TrackingState
        };

        foreach (Joint joint in skeleton.Joints)
        {
            var filteredJoint = new FilteredJoint
            {
                JointType = joint.JointType
            };

            var filteredPositions = new double[3];
            for (var i = 0; i < 3; i++)
            {
                double measurement = i switch
                {
                    0 => joint.Position.X,
                    1 => joint.Position.Y,
                    2 => joint.Position.Z,
                    _ => 0
                };
                filteredPositions[i] = _kalmanFilters[joint.JointType][i].Update(measurement);
            }

            filteredJoint.Position = new SkeletonPoint
            {
                X = (float)filteredPositions[0],
                Y = (float)filteredPositions[1],
                Z = (float)filteredPositions[2]
            };

            filteredSkeleton.Joints.Add(filteredJoint);
        }

        return filteredSkeleton;
    }
}
