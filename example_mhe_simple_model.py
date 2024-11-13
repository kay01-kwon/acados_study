import numpy as np
from scipy.integrate import odeint
from solver.mhe_solver_for_simple_system import MheSolverForSimpleSystem
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

mass_true = 1
def dynamics_of_simple_system(x, t, F):
    A = np.array([[0, 1], [0, 0]])
    B = np.array([0, 1])
    u = F/mass_true
    dxdt = A.dot(x) + B*u
    return dxdt

if __name__ == "__main__":

    Tf = 2.0
    dt = 0.01
    t = np.arange(0, Tf + dt, dt)
    N = t.shape[0]

    x0 = np.array([1, 0])

    x = np.zeros((N,2))
    x[0,:] = x0

    F_vec = np.zeros((N,1))

    # Standard deviation of state noise
    v_stds = [0.01, 0.01]

    # Control PD gain
    Kp = 1
    Kd = 0.03

    # Simulate the plant
    for i in range(N-1):
        tspan = [t[i], t[i+1]]
        x[i+1,:] = odeint(dynamics_of_simple_system, x0, tspan, args=(F_vec[i], ))[-1]
        F_vec[i+1] = -Kp*x[i+1,0] - Kd*x[i+1,1]
        x0 = x[i+1,:]

    dim_x = x.shape[1]


    x_noise = np.zeros((N, dim_x))
    # Add noise to the state
    for i in range(N):
        x_noise[i,:] = x[i,:] + np.transpose(np.diag(v_stds) @ np.random.standard_normal((2, 1)))

    # Horizon for MHE
    N_horizon = 40
    time_horizon = dt*N_horizon

    # Weight for Arrival cost
    Q0 = np.diag([1e-3, 1e-3, 1])

    # Weight for noise
    Q = np.diag([1e-3, 1e-3, 1e-3])

    # Weight for State
    R = 1/dt*np.diag([1, 1])

    # Initial guess about the augmented state
    x0_bar = np.array([x0[0], x0[1], 0.1])

    acados_solver_mhe = MheSolverForSimpleSystem(N_horizon, time_horizon, Q0, Q, R).get_ocp_solver()

    x_est = np.zeros((N, dim_x))
    x_est[0,:] = x0_bar[:2]
    m_est = np.zeros((N,1))
    m_true = mass_true*np.ones((N,1))
    noise_est = np.zeros((N,3))

    for i in range(N_horizon):
        acados_solver_mhe.set(i, "x", x0_bar)

    yref_0 = np.zeros((8,))
    yref = np.zeros((5,))

    plt.rcParams["lines.linewidth"] = 4
    plt.rcParams["axes.grid"] = True
    plt.rcParams['font.size'] = 20

    fig, ax = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(30, 14)
    line_pos_state, = ax[0].plot(t[0], x_noise[0,0], 
                                 label="Pos state", marker="*")
    
    line_pos_state_est, = ax[0].plot(t[0], x_est[0,0], label="Pos est")
    ax[0].title.set_text("Position - time (s)")
    ax[0].set(xlim=(t[0], t[-1]), ylim=(-1, 1), xlabel="Time", ylabel="pos")
    ax[0].legend()

    line_vel_state, = ax[1].plot(t[0], x_noise[0,1], 
                                 label="Vel state", marker="*")
    line_vel_state_est, = ax[1].plot(t[0], x_est[0,1], label="Vel est")
    ax[1].title.set_text("Velocity - time (s)")
    ax[1].set(xlim=(t[0], t[-1]), ylim=(-1, 1), xlabel="Time", ylabel="pos")
    ax[1].legend()

    line_mass_true, = ax[2].plot(t[0], m_true[0], label="True mass")
    line_mass_est, = ax[2].plot(t[0], m_est[0], label="Est mass (MHE)")
    ax[2].title.set_text("Mass - time (s)")
    ax[2].set(xlim=(t[0], t[-1]), ylim=(0.5, 1.5), xlabel="Time", ylabel="Mass")
    ax[2].legend()


    def update(frame):
        line_pos_state.set_data(t[:frame], x_noise[:frame, 0])
        line_pos_state_est.set_data(t[:frame], x_est[:frame, 0])

        line_vel_state.set_data(t[:frame], x_noise[:frame, 1])
        line_vel_state_est.set_data(t[:frame], x_est[:frame, 1])

        line_mass_true.set_data(t[:frame], m_true[:frame])
        line_mass_est.set_data(t[:frame], m_est[:frame])
        # print(x_est[:frame, 0])
        return line_pos_state, line_pos_state_est, line_vel_state, line_vel_state_est, 
        line_mass_true, line_mass_est

    for i in range(N - N_horizon):
        yref_0[:2] = x_noise[i,:]
        yref_0[5:] = x0_bar
        acados_solver_mhe.set(0,"yref", yref_0)
        acados_solver_mhe.set(0,"p",F_vec[i])

        for j in range(i + 1, N_horizon + i):
            yref[:2] = x_noise[j,:]
            # print(j-i, yref)
            acados_solver_mhe.set(j-i, "yref", yref)
            acados_solver_mhe.set(j-i, "p", F_vec[j])

        acados_solver_mhe.solve()

        x_augmented = acados_solver_mhe.get(1, "x")
        x0_bar = x_augmented
        x_est[i, :] = x_augmented[0:2]
        # print(x_est[:,0])
        m_est[i, :] = x_augmented[2]
        noise_est[i, :] = acados_solver_mhe.get(1, "u")

ani = animation.FuncAnimation(fig = fig, func = update, 
                              frames = N - N_horizon, interval = 100)
# plt.show()
ani.save("mhe_simple_system.gif", writer='pillow', fps = 30)


    # plt.figure(1)
    # plt.subplot(3,1,1)
    # plt.plot(t[:N-N_horizon], x_noise[:N-N_horizon,0], marker="*", label="Meas_pos")
    # plt.plot(t[:N-N_horizon], x_est[:N-N_horizon,0], label="Est_pos")
    #
    # plt.subplot(3,1,2)
    # plt.plot(t[:N-N_horizon], x_noise[:N-N_horizon,1], marker='*', label="Meas_vel")
    # plt.plot(t[:N-N_horizon],x_est[:N-N_horizon,1], label="Est_vel")
    #
    # plt.subplot(3,1,3)
    # plt.plot(t[:N-N_horizon],m_true[:N-N_horizon], label="True_mass")
    # plt.plot(t[:N-N_horizon], m_est[:N-N_horizon], label="Est_mass")

    # plt.show()