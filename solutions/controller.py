import numpy as np
from systems import BicycleModel
from trajectory import compute_circle_start_on_circle, wrap_circular_value
import matplotlib.pyplot as plt
import os
from scipy.linalg import solve_continuous_are, inv


class Controller:
    def __init__(self):
        self.e_perp = 0.0
        self.integral_errors = np.zeros(4)

    def ctrl_linear(self,
                    state:np.ndarray,
                    state_d:np.ndarray,
                    params_robot:dict,
                    params_ctrl:dict,
                    part_pb: str) -> np.ndarray:

        p_I_x, p_I_y, theta, v_B_x, v_B_y, omega = state
        p_d_I_x, p_d_I_y, theta_d, v_d_I_x, v_d_I_y, omega_d = state_d

        p_I = np.array([p_I_x, p_I_y]) # Positions in the inertial frame.
        p_d_I = np.array([p_d_I_x, p_d_I_y]) # Desired positions in the inertial frame.
        v_d_I = np.array([v_d_I_x, v_d_I_y]) # Desired velocities in the inertial frame.

        p_err_I = p_I - p_d_I # Position error in the inertial frame.
        R_d = np.array([[np.cos(theta_d), -np.sin(theta_d)],
                        [np.sin(theta_d), np.cos(theta_d)]])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        perr_B = R.T @ p_err_I
        perr_d_B = R_d.T @ p_err_I
        v_d_B = R_d.T @ v_d_I
        R_error = R_d.T @ R

        # e1.
        # self.e_perp = perr_B[1]
        # e1_dot.
        theta_err = np.arctan2(R_error[1][0], R_error[0][0])
        e_perp_dot = np.sin(theta_err) * v_B_x + np.cos(theta_err) * v_B_y # Eq 34 in MAGIC VFM paper
        self.e_perp = self.e_perp + e_perp_dot * DT
        # e2.
        # e2_dot.
        omega_err = omega - omega_d
        
        e = np.array([self.e_perp, e_perp_dot, theta_err, omega_err])

        if part_pb == 'c':
            K = np.array([params_ctrl['K_E'], params_ctrl['K_E_DOT'], params_ctrl['K_THETA'], params_ctrl['K_OMEGA']])
            u_steering = -K @ e

        if part_pb == 'd':
            K = np.array([params_ctrl['K_E'], params_ctrl['K_E_DOT'], params_ctrl['K_THETA'], params_ctrl['K_OMEGA']])
            K_I = np.array([params_ctrl['K_I_E'], params_ctrl['K_I_E_DOT'], params_ctrl['K_I_THETA'], params_ctrl['K_I_OMEGA']])
            u_steering = -K @ e - K_I @ self.integral_errors
            dintegral_errors = e.copy()
            self.integral_errors += dintegral_errors * DT
            for i in range(len(self.integral_errors)):
                self.integral_errors[i] = np.clip(self.integral_errors[i], -params_ctrl['MAX_INTEGRAL'], params_ctrl['MAX_INTEGRAL'])

        if part_pb == 'e':
            A = np.array([
                [0, 1, 0, 0],
                [0, -params["Cy"]/(params["mass"]*v_B_x), params["Cy"]/params["mass"], 0],
                [0, 0, 0, 1],
                [0, 0, 0, -params["L"]**2 * params["Cy"] / (2 * params["Iz"] * v_B_x)]
                ])

            B = np.array([
                [0],
                [params["Cy"]/params["mass"]],
                [0],
                [params["Cy"]*params["L"] / (2 * params["Iz"])]
                ])
            Q = np.array([[3, 0, 0, 0],
                            [0, 0.1, 0, 0],
                            [0, 0, 3, 0],
                            [0, 0, 0, 0.1]])
            R = np.array([[5]])
            P = solve_continuous_are(A, B, Q, R)
            K = inv(R) @ B.T @ P
            u_steering = -K[0] @ e

        if part_pb == 'f':
            K = np.array([params_ctrl['K_E'], params_ctrl['K_E_DOT'], params_ctrl['K_THETA'], params_ctrl['K_OMEGA']])
            radius_traj = params_traj['R']
            u_ff = params_robot['L']/radius_traj * (1 - params_ctrl['K_THETA']) + params_ctrl['K_THETA'] * params["mass"] * v_B_x**2 / (params["Cy"] * radius_traj)
            u_steering = -K @ e + u_ff

        u_steering = np.clip(u_steering, -params['max_steering'], params['max_steering'])

        b = 1/params['tau']
        a = -1/params['tau']
        KP = 1.0
        u_v = -1/b * (KP * (v_B_x - v_d_B[0]) + a * v_d_B[0])
        u_v = np.clip(u_v, -params['max_vel'], params['max_vel'])

        # Debug params.
        outputs = {
            'e_perp': self.e_perp,
            'e_perp_dot': e_perp_dot,
            'theta_err': theta_err,
            'omega_err': omega_err,
            'v_d_B_x': v_d_B[0],
            'v_B_x': v_B_x,
            'v_d_B_y': v_d_B[1],
            'v_B_y': v_B_y,
            'u_steering': u_steering,
            'u_v': u_v,
        }

        return np.array([u_v, u_steering]), outputs

# Simulation parameters
N = 5000
DT = 0.01
PB_PART = 'f'

# Model params.
params = {"dt": DT,
        "tau": 0.1,
        'L': 2.9718,
        'Cy': 70000.00,
        'mass': 815.11,
        'Iz': 800.0,
        'tau': 0.1,
        'max_steering': np.deg2rad(17),
        'max_vel': 300.0,
}

params_ctrl = {
    'K_E': 0.97459667,
    'K_E_DOT': 0.7279311,
    'K_THETA': 4.33612155,
    'K_OMEGA': 0.0,
    'K_I_E': 0.8,
    'K_I_E_DOT': 0.0,
    'K_I_THETA': 0.0,
    'K_I_OMEGA': 0.0,
    'MAX_INTEGRAL': 10.0,
}

params_traj = {
    'v_desired': 40.0,
    'R': 200,
}

# Set the model.
car = BicycleModel(params)

# Des. Traj.
des_traj_array = np.empty((N, 6))
theta_d_previous = 0.0
e_perp_sum = 0.0

# Outputs dict.
output_dict_list = []

action_list = np.empty((N, 2))

angle = 0.0
state = np.array([0.0, 0.0, 0.0, params_traj['v_desired'], 0.0, 0.0])
state0 = state.copy()
state_array = np.empty((N, 6))

# controller
controller = Controller()

for i in range(N):
    # Compute desired trajectory.
    x_d_I, y_d_I, _, vx_d_I, vy_d_I, _, angle = compute_circle_start_on_circle(angle=angle,
                                                                               dt=DT,
                                                                               v_desired=params_traj['v_desired'],
                                                                               initial_state_I=[state0[0], state0[1], state0[2]],
                                                                               R=params_traj['R'])
    theta_d = np.arctan2(vy_d_I, vx_d_I)
    omega_d = wrap_circular_value((theta_d - theta_d_previous)/DT)
    theta_d_previous = theta_d
    state_d = np.array([x_d_I, y_d_I, theta_d, vx_d_I, vy_d_I, omega_d])

    action, outputs = controller.ctrl_linear(state=state,
                                        state_d=state_d,
                                        params_robot=params,
                                        params_ctrl=params_ctrl,
                                        part_pb=PB_PART)

    # Propagate.
    next_state = car.dynamics(state, action)
    state = next_state.copy()
    
    # Save data.
    des_traj_array[i, :] = state_d
    state_array[i, :] = state
    output_dict_list.append(outputs)
    action_list[i, :] = action
    

fig, ax = plt.subplots(2, 2, figsize=(4, 4))
plt.suptitle('Desired Trajectory')
ax[0, 0].plot(des_traj_array[:, 0], des_traj_array[:, 1], 'r-', label='xy des')
ax[0, 0].plot(des_traj_array[0, 0], des_traj_array[0, 1], 'ro')
ax[0, 0].plot(des_traj_array[-1, 0], des_traj_array[-1, 1], 'rx')
ax[0, 1].plot(des_traj_array[:, 2], 'g-', label='theta des')
ax[1, 0].plot(des_traj_array[:, 3], 'r-', label='v_x des (Inertial)')
ax[1, 0].plot(des_traj_array[:, 4], 'b-', label='v_y des (Inertial)')
ax[1 ,1].plot(des_traj_array[:, 5], 'g-', label='omega des')
for i in range(2):
    for j in range(2):
        ax[i, j].legend(loc='upper right', fontsize=8)
plt.tight_layout()

plt.figure(figsize=(3, 3))
plt.plot(des_traj_array[:, 0], des_traj_array[:, 1], label='Desired')
plt.plot(des_traj_array[0, 0], des_traj_array[0, 1], 'ro')
plt.plot(state_array[:, 0], state_array[:, 1], label='Actual')
plt.axis('equal')
plt.legend()
plt.title('Performance')

e_perp_array = np.array([output_dict['e_perp'] for output_dict in output_dict_list])
e_perp_dot_array = np.array([output_dict['e_perp_dot'] for output_dict in output_dict_list])
theta_err_array = np.array([output_dict['theta_err'] for output_dict in output_dict_list])
omega_err_array = np.array([output_dict['omega_err'] for output_dict in output_dict_list])

fig, ax = plt.subplots(1, 6, figsize=(16, 2))
ax[0].plot(e_perp_array, label='e_perp')
ax[0].legend()
ax[1].plot(e_perp_dot_array, label='e_perp_dot')
ax[1].legend()
ax[2].plot(theta_err_array, label='theta_err')
ax[2].legend()
ax[3].plot(omega_err_array, label='omega_err')
ax[3].legend()
ax[4].plot(des_traj_array[:, 0], des_traj_array[:, 1])
ax[4].plot(des_traj_array[0, 0], des_traj_array[0, 1], 'ro')
ax[4].plot(state_array[:, 0], state_array[:, 1])
ax[4].set_ylabel('Y [m]')
ax[4].set_xlabel('X [m]')
# Velocities.
v_d_x = np.array([output_dict['v_d_B_x'] for output_dict in output_dict_list])
v_d_y = np.array([output_dict['v_d_B_y'] for output_dict in output_dict_list])
ax[5].plot(state_array[:, 3], label='$v_x^B$')
ax[5].plot(v_d_x, label='$v_{x,d}^B$')
ax[5].plot(state_array[:, 4], label='$v_y^B$')
ax[5].plot(v_d_y, label='$v_{y,d}^B$')
plt.tight_layout()
plt.legend()

figure_folder = 'figures/'
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)
plt.savefig(figure_folder + 'pb_' + PB_PART + '.pdf')

plt.show()
