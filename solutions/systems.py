import numpy as np
import matplotlib.pyplot as plt

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi

class BicycleModel:
    xdim: int = 6
    xnames = ["x_I", "y_I", "theta", "vx_B", "vy_B", "omega"]
    udim: int = 2
    unames = ["u_v", "u_steering"]

    def __init__(self, params: dict):
        self.dt = params['dt']
        self.params = params

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, theta, vx_B, vy_B, omega = state
        u_v, u_steering = action
        
        # Params.
        L = self.params['L']
        mass = self.params['mass']
        Cy = self.params['Cy']
        Iz = self.params['Iz']
        tau = self.params['tau']

        vx_dot = 1/tau * (u_v - vx_B)
        vy_dot = - Cy/(mass * vx_B) * vy_B - vx_B * omega + Cy/mass * u_steering
        omega_dot = -L**2 * Cy/(2 * Iz * vx_B) * omega + L * Cy/(2*Iz) * u_steering
    
        vx_B = vx_B + self.dt * vx_dot
        vy_B = vy_B + self.dt * vy_dot
        omega = omega + self.dt * omega_dot

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        v_B = np.array([vx_B, vy_B])
        v_I = R @ v_B
        x = x + self.dt * v_I[0]
        y = y + self.dt * v_I[1]
        theta = theta + self.dt * omega
        # Note, you can also pick a different representation for theta, then you don't need to wrap the angle.
        theta = wrap_circular_value(theta)

        next_state = np.array([x, y, theta, vx_B, vy_B, omega])

        return next_state
