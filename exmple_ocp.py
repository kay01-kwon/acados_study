from acados_template import AcadosOcp, AcadosOcpSolver
from model import dynamics_ode
from utils import plot_util
import numpy as np

def run():
    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = dynamics_ode()
    ocp.model = model

    Tf = 1.0
    N = 20
    nx = model.x.rows()
    nu = model.u.rows()

    # Set the number of shooting intervals
    ocp.dims.N = N

    # Set cost
    Q_mat = 2*np.diag([100, 100])
    R_mat = 2*np.diag([10])

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Umax = 10
    ocp.constraints.lbu = np.array([-Umax])
    ocp.constraints.ubu = np.array([Umax])
    ocp.constraints.idxbu = np.array([0, 1])