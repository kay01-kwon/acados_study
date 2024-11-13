from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from model import system_1d
import scipy.linalg
from utils import plot_util
import numpy as np
import matplotlib.pyplot as plt

# Initial state
X0 = np.array([0.0, 0.0])
T_horizon = 2.0

def create_ocp_solver() -> AcadosOcp:
    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = system_1d.export_dynamics_ode_model()
    ocp.model = model

    N = 50
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    ny_e = nx

    # Set the number of shooting intervals
    ocp.dims.N = N

    # OCP objective function
    # Set cost
    # cost Q: x, v
    Q_mat = 2*np.diag([1000, 700])
    # cost R: u
    R_mat = 2*5*1e-1

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vx[0,0] = 1.0
    ocp.cost.Vx[1,1] = 1.0
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e,))
    ocp.cost.W_e = Q_mat


    Umax = 10
    ocp.constraints.lbu = np.array([-Umax])
    ocp.constraints.ubu = np.array([Umax])
    # ocp.constraints.lbx = np.array([-100, -1, -Umax])
    # ocp.constraints.ubx = np.array([100, 1, Umax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp

def closed_loop_simulation():

    # Create solvers
    ocp = create_ocp_solver()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    N_horizon = acados_ocp_solver.N

    # Prepare for simulation
    Nsim = 200
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim,nu))

    xcurrent = X0
    simX[0,:] = xcurrent

    y_ref = np.array([1, 0, 0])
    y_ref_N = np.array([1, 0])

    # Initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", 0.0*np.ones(xcurrent.shape))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",np.zeros((nu,)))

    # Closed loop
    for i in range(Nsim):

        for j in range(N_horizon):
            acados_ocp_solver.set(j,"y_ref", y_ref)
        acados_ocp_solver.set(N_horizon, "y_ref", y_ref_N)

        simU[i,:] = acados_ocp_solver.solve_for_x0(xcurrent)
        xcurrent = acados_integrator.simulate(xcurrent, simU[i,:])
        simX[i + 1,:] = xcurrent

    plt.plot(np.linspace(0, T_horizon/N_horizon*Nsim, Nsim+1),simX[:,0])
    plt.xlabel("Time")
    plt.ylabel("x [m]")

    # plt.plot(simU)

    plt.show()

if __name__ == "__main__":
    closed_loop_simulation()