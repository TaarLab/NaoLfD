namespace KinectSkeleton;


public class KalmanFilter
{
    private double Q; // Process noise covariance
    private double R; // Measurement noise covariance
    private double P; // Estimation error covariance
    private double X; // Estimated value
    private double K; // Kalman gain

    public KalmanFilter(double processNoise, double measurementNoise, double initialValue)
    {
        Q = processNoise;
        R = measurementNoise;
        P = 1.0; // Initial estimation error covariance
        X = initialValue; // Initial state estimate
    }

    public double Update(double measurement)
    {
        // Prediction update
        P += Q;

        // Measurement update
        K = P / (P + R);
        X += K * (measurement - X);
        P *= 1 - K;

        return X;
    }
}