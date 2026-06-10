import numpy as np

class KobukiVelocitySmoother:
    """
    Non‑vectorized NumPy version of the Kobuki velocity smoother.
    Matches the behavior of the Kobuki C++ smoother:
    - clamp to speed limits
    - ramp toward target using accel/decel limits
    """

    def __init__(
        self,
        speed_lim_v=0.5,     # max linear speed (m/s)
        speed_lim_w=1.0,     # max angular speed (rad/s)
        accel_lim_v=0.3,     # max linear accel (m/s^2)
        accel_lim_w=0.5,     # max angular accel (rad/s^2)
        dt=1/20.0            # control timestep
    ):
        self.speed_lim_v = speed_lim_v
        self.speed_lim_w = speed_lim_w
        self.accel_lim_v = accel_lim_v
        self.accel_lim_w = accel_lim_w
        self.dt = dt

        # current smoothed velocity [v, w]
        self.current = np.zeros((1, 2), dtype=np.float32)

    def step(self, action):
        """
        action: np.array([v_cmd, w_cmd]) in [-1, 1]
        returns: np.array([v_smooth, w_smooth])
        """

        # 1) Scale agent outputs to real robot limits
        target_v = action[0, 0] * self.speed_lim_v
        target_w = action[0, 1] * self.speed_lim_w

        # 2) Compute increments
        dv = target_v - self.current[0, 0]
        dw = target_w - self.current[0, 1]

        max_dv = self.accel_lim_v * self.dt
        max_dw = self.accel_lim_w * self.dt

        # 3) Clamp increments
        dv = np.clip(dv, -max_dv, max_dv)
        dw = np.clip(dw, -max_dw, max_dw)

        # 4) Update smoothed velocity
        self.current[0, 0] += dv
        self.current[0, 1] += dw

        return self.current.copy()

    def reset(self):
        self.current[0,:] = 0.0
