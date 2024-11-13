import matplotlib.pyplot as plt
import numpy as np


def plot_result(t, u_max, U, X_true):
    """
    Params
    :param t: Time values
    :param u_max: maximum absolute control input value
    :param U: Control input vectors (N_sim-1, nu)
    :param X_true: State vectors (N_sim, nx)
    """

    nx = X_true.shape[1]
    fig, axes = plt.subplots(nx+1, 1, sharex=True, sharey=True)

    for i in range(nx):
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()

    axes[-1].step( t, np.append( [U[0], U] ) )

    plt.show()