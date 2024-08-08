import numpy as np
import scipy.linalg
from acados_template import AcadosOcpSolver, AcadosOcp
from model.mhe_simple_model import MheSimpleModel
import casadi as cs

class MheSolverForSimpleSystem:
    def __init__(self, N, tf, Q, Q0, R):
        '''
        :param N: Number of shooting nodes
        :param tf: prediction horizon
        :param Q: Weight matrix for the observed state variables
        :param Q0: Terminal weight matrix for state variables
        :param R: Weight matrix for the noise of the observed state variables
        '''

        self.ocp_mhe = AcadosOcp()
        self.acados_mhe_solver = []

        self.ocp_mhe.model = MheSimpleModel().get_acados_model()

        self.tf = tf

        # Dimension info
        self.nx = self.ocp_mhe.model.x.rows()
        self.nu = self.ocp_mhe.model.u.rows()

        self.ny_0 = 2 + 2*self.nx         # state, noise, Arrival cost
        self.ny = 2 + self.nx                     # state, noise
        self.ny_e = 0

        self.nparam = self.ocp_mhe.model.p.rows()
        # Set the number of shooting nodes

        self.ocp_mhe.dims.N = N

        self.ocp_mhe.cost.cost_type = 'LINEAR_LS'       # Linear least square
        self.ocp_mhe.cost.cost_type_0 = 'LINEAR_LS'     # Linear least square

        self.ocp_mhe.parameter_values = np.zeros((self.nparam,))

        self.set_ocp_cost(Q, Q0, R)

        self.set_ocp_solver()

        # set arrival cost weighting matrix
        self.acados_mhe_solver.cost_set(0, "W", scipy.linalg.block_diag(R, Q, Q0))


    def set_ocp_cost(self, Q, Q0, R):
        # Setup weight for cost

        # 1. Initial stage
        self.ocp_mhe.cost.Vx_0 = np.zeros((self.ny_0, self.nx))
        self.ocp_mhe.cost.Vx_0[:self.nx,:] = np.eye(self.nx)
        self.ocp_mhe.cost.Vx_0[self.nx:2*self.nx,:] = np.eye(self.nx)

        self.ocp_mhe.cost.Vu_0 = np.zeros((self.ny_0, self.nu))
        self.ocp_mhe.cost.Vu_0[self.nx:self.nx+self.nu,:] = np.eye(self.nu)

        self.ocp_mhe.cost.W_0 = scipy.linalg.block_diag(R, Q, Q0)

        self.ocp_mhe.cost.yref_0 = np.zeros((self.ny_0,))

        # 2. Intermidiate stage (Lagrange term)
        self.ocp_mhe.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp_mhe.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.ocp_mhe.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp_mhe.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu)

        # 4. Weight for state cost
        self.ocp_mhe.cost.W = scipy.linalg.block_diag(R, Q)

        self.ocp_mhe.cost.yref = np.zeros((self.ny,))

    def set_ocp_solver(self):

        # Set QP solver
        self.ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        self.ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp_mhe.solver_options.integrator_type = 'ERK'

        self.ocp_mhe.solver_options.nlp_solver_type = 'SQP'
        self.ocp_mhe.solver_options.nlp_solver_max_iter = 200

        # Set prediction horizon
        self.ocp_mhe.solver_options.tf = self.tf

        self.acados_mhe_solver = AcadosOcpSolver(self.ocp_mhe)

    def get_ocp_solver(self):

        return self.acados_mhe_solver