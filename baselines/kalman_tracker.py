import numpy as np

class KalmanFilterTracker:
    """
    Standard linear Kalman Filter for 1D range and velocity tracking.
    State vector: x = [range, velocity]^T
    Observation: z = [range_meas, velocity_meas]^T
    """
    def __init__(self, dt: float, process_noise_std: float = 0.5, meas_noise_std: float = 2.0):
        self.dt = dt
        # State transition model
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        # Observation model
        self.H = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # Process noise covariance (constant velocity model)
        q = process_noise_std**2
        self.Q = np.array([
            [q * (dt**3)/3, q * (dt**2)/2],
            [q * (dt**2)/2, q * dt]
        ])
        
        # Measurement noise covariance
        r = meas_noise_std**2
        self.R = np.array([
            [r, 0.0],
            [0.0, r]
        ])
        
        # Initial state and covariance
        self.x = np.zeros(2)
        self.P = np.eye(2) * 100.0
        self.is_initialized = False

    def initialize(self, z: np.ndarray):
        self.x = z.copy()
        self.P = np.eye(2) * 100.0
        self.is_initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
    def step(self, z: np.ndarray) -> np.ndarray:
        """
        Runs one step of the filter.
        Args:
            z: Measurement vector [range, velocity]
        Returns:
            Filtered state vector [range, velocity]
        """
        if not self.is_initialized:
            self.initialize(z)
            return self.x
            
        self.predict()
        self.update(z)
        return self.x
