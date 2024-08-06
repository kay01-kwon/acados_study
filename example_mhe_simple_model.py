from model.mhe_simple_model import MheSimpleModel
import numpy as np
from scipy.integrate import odeint
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

    Tf = 1
    dt = 0.01
    t = np.arange(0, Tf + dt, dt)

    F = 1.0

    # x = odeint((lambda t,x :dynamics_of_simple_system(x, t, F)), x0, t)

    x = odeint(dynamics_of_simple_system, x0, t, args=(F,))

    plt.plot(t, x[:, 0], label="position")
    plt.title("Position - time")

    plt.show()
