import math
import numpy as np

def wrap_circular_value(input_value: float) -> float:
    return (input_value + np.pi) % (2*np.pi) - np.pi

def compute_circle_start_on_circle(angle: float,
                                   dt: float,
                                   v_desired: float,
                                   initial_state_I: np.ndarray,
                                   R:float) -> np.ndarray:
    omega = v_desired / R
    angle = angle + dt * omega
    angle = wrap_circular_value(angle)

    x0, y0, th0 = initial_state_I[0], initial_state_I[1], initial_state_I[2]
    x = x0 - R * math.sin(th0) + R * math.sin((angle + th0))
    y = y0 + R * math.cos(th0) - R * math.cos((angle + th0))
    z = 0.0
    x_dot = R * math.cos(angle + th0) * omega
    y_dot = R * math.sin(angle + th0) * omega
    z_dot = 0.0

    return x, y, z, x_dot, y_dot, z_dot, angle
