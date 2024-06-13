from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from model import system_2d
import scipy.linalg
from utils import plot_util
import numpy as np
import matplotlib.pyplot as plt

# Initial state
X0 = np.array([0.0, 0.0, 0.0, 0.0])
T_horizon = 1.0

def create_ocp_solver() -> AcadosOcp:
    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = system_2d.export_dynamics_ode_model()
    ocp.model = model

    N = 20
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    ny_e = nx

    # Set the number of shooting intervals
    ocp.dims.N = N

    # OCP objective function
    # Set cost
    # cost Q: p_x,p_y, v_x, v_y
    Q_mat = 2*np.diag([10, 10, 7, 7])
    # cost R: u
    R_mat = 2*5*np.diag([1e-1, 1e-1])

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vx[0,0] = 1.0
    ocp.cost.Vx[1,1] = 1.0
    ocp.cost.Vx[2, 2] = 1.0
    ocp.cost.Vx[3, 3] = 1.0

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e,))
    ocp.cost.W_e = Q_mat

    Umax = 10
    ocp.constraints.lbu = np.array([-Umax, -Umax])
    ocp.constraints.ubu = np.array([Umax, Umax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.constraint_type = 'bgp'
    h_expr = getCBF(model.x[0:2], model.x[2:4], model.u)
    ocp.model.con_h_expr = h_expr
    ocp.constraints.lh = np.array([0])
    ocp.constraints.uh = np.array([1e15])

    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp

def getCBF(p, v, u):
    Lf2h = 2*(v[0]**2 + v[1]**2)
    Lfghf = 2*( (p[0]-2)*u[0] + (p[1]-2)*u[1] )
    Lfh = 2*( (p[0]-2)*v[0] + (p[1]-2)*v[1])
    h = (p[0]-2)**2 + (p[1]-2)**2 - 1

    k1 = 8
    k2 = 64
    return Lf2h + Lfghf + k1*Lfh + k2*h

def closed_loop_simulation():

    # Create solvers
    ocp = create_ocp_solver()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    N_horizon = acados_ocp_solver.N

    # Prepare for simulation
    Nsim = 100
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim,nu))
    h_values = np.zeros((Nsim,1))

    xcurrent = X0
    simX[0,:] = xcurrent


    # p_x, p_y, v_x, v_y, u_x, u_y
    y_ref = np.array([5, 5, 0, 0, 0, 0])
    y_ref_N = np.array([5, 5, 0, 0])

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
        # h_values[i] = acados_ocp_solver.get(j,'h')

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.linspace(0, T_horizon/N_horizon*Nsim, Nsim + 1),simX[:,0], linewidth=4)
    plt.plot(np.linspace(0, T_horizon/N_horizon*Nsim, Nsim + 1),simX[:,1], linewidth=4)
    plt.xlabel("Time",fontsize=32)
    plt.ylabel("x [m]",fontsize=32)
    plt.xticks([0,2,4,6,8],fontsize=32)
    plt.yticks([0, 1.0, 2.0, 3.0, 5.0],fontsize=32)
    plt.grid('true')

    plt.subplot(1,2,2)
    plt.plot(np.linspace(0, T_horizon / N_horizon * Nsim, Nsim), simU[:, 0], linewidth=4)
    plt.plot(np.linspace(0, T_horizon / N_horizon * Nsim, Nsim), simU[:, 1], linewidth=4)
    plt.xlabel("Time",fontsize=32)
    plt.ylabel("Control input", fontsize=32)
    plt.xticks([0,2,4,6,8],fontsize=32)
    plt.yticks([-2, 0, 2, 4, 6, 8 ,10],fontsize=32)
    plt.grid('true')

    plt.figure()
    plt.plot(simX[:,0], simX[:,1], linewidth=4)


    plt.show()

if __name__ == "__main__":
    closed_loop_simulation()