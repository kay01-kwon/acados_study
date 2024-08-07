import numpy as np
from scipy.integrate import odeint
from solver.mhe_solver_for_simple_system import MheSolverForSimpleSystem
from model.mhe_simple_model import MheSimpleModel
import matplotlib.pyplot as plt

def dynamics_of_simple_system(x, t, F):
    mass = 1
    A = np.array([[0, 1], [0, 0]])
    B = np.array([0, 1])
    u = F/mass
    dxdt = A.dot(x) + B*u

    return dxdt


if __name__ == "__main__":

    x0 = np.array([0, 0])

    Tf = 1.0
    dt = 0.01
    t = np.arange(0, Tf + dt, dt)

    F = 10.0

    v_stds = [0., 0.]

    x = odeint(dynamics_of_simple_system, x0, t, args=(F,))

    N = x.shape[0]
    dim_x = x.shape[1]


    x_noise = np.zeros((N, dim_x))

    for i in range(N):
        x_noise[i,:] = x[i,:] + np.transpose(np.diag(v_stds) @ np.random.standard_normal((2, 1)))

    # plt.plot(t, x_noise[0,:], label="noisy signal")
    # plt.plot(t, x_noise[:,1], label="noisy signal")


    time_horizon = Tf

    Q0 = np.diag([0.1, 0.1, 0.01])
    Q = np.diag([0.01, 0.01, 0.01])
    R = np.diag([100, 100])

    acados_solver_mhe = MheSolverForSimpleSystem(N, time_horizon, Q, Q0, R).get_ocp_solver()

    x0_bar = np.array([0, 0, 0.8])

    yref_0 = np.zeros((8,))

    yref_0[:2] = x_noise[0,:]
    yref_0[5:] = x0_bar

    acados_solver_mhe.set(0, "yref", yref_0)
    acados_solver_mhe.set(0, "p", F)

    # set initial guess to x0_bar
    acados_solver_mhe.set(0, "x", x0_bar)

    yref = np.zeros((5,))

    for j in range(1, N):
        yref[:2] = x_noise[j, :]
        acados_solver_mhe.set(j, "yref", yref)
        acados_solver_mhe.set(j, "p", F)

        # set initial guess to x0_bar
        acados_solver_mhe.set(j, "x", x0_bar)

    # set initial guess to x0_bar
    acados_solver_mhe.set(N, "x", x0_bar)

    # solve mhe problem
    status = acados_solver_mhe.solve()

    x_est = np.zeros((N, dim_x))
    m_est = np.zeros((N,1))
    noise_est = np.zeros((N,3))

    # get solution
    for i in range(N):
        x_augmented = acados_solver_mhe.get(i, "x")

        x_est[i, :] = x_augmented[0:2]
        m_est[i, :] = x_augmented[2]
        noise_est[i, :] = acados_solver_mhe.get(i, "u")


    # plt.plot(t, x_est[:,0])
    # plt.plot(t, x[:,0])
    plt.plot(t, m_est)

    plt.show()