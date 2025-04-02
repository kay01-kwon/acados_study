import numpy as np
import scipy.linalg
from acados_template import AcadosOcpSolver, AcadosOcp
from model.mhe_disturbance_estimator import DisturbanceEstimator
import casadi as cs

class Solver_Distubance_Estimator():
    def __init__(self, N, tf, Q0, Q, R, Parameter):
        '''
        :param N: Number of shooting nodes
        :param tf: prediction horizon
        :param Q: Weight matrix for the observed state variables
        :param Q0: Terminal weight matrix for state variables
        :param R: Weight matrix for the noise of the observed state variables
        '''

        self.ocp_mhe = AcadosOcp()
        self.acados_mhe_solver = []

        self.ocp_mhe.model = DisturbanceEstimator(Parameter).get_acados_model()

        self.tf = tf

        # Dimension info
        self.nx_aug = self.ocp_mhe.model.x.rows()
        self.nx = self.nx_aug - 1
        self.nu = self.ocp_mhe.model.u.rows()

        self.x = self.ocp_mhe.model.x
        self.u = self.ocp_mhe.model.u

        self.ny_0 = R.shape[0] + Q.shape[0] + Q0.shape[0]         # state, noise, Arrival cost
        self.ny = R.shape[0] + Q.shape[0]                     # state, noise
        self.ny_e = 0

        self.nparam = self.ocp_mhe.model.p.rows()
        # Set the number of shooting nodes

        self.ocp_mhe.dims.N = N

        # self.ocp_mhe.cost.cost_type = 'LINEAR_LS'       # Linear least square
        # self.ocp_mhe.cost.cost_type_0 = 'LINEAR_LS'     # Linear least square

        # set cost type
        self.ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
        self.ocp_mhe.cost.cost_type_e = 'LINEAR_LS'
        self.ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'

        self.ocp_mhe.parameter_values = np.zeros((self.nparam,))

        self.set_ocp_cost(Q0, Q, R)

        self.set_ocp_solver()

        # set arrival cost weighting matrix
        self.acados_mhe_solver.cost_set(0, "W", scipy.linalg.block_diag(R, Q, Q0))


    def set_ocp_cost(self, Q0, Q, R):
        # Setup weight for cost

        # 1. Initial stage
        # self.ocp_mhe.cost.Vx_0 = np.zeros((self.ny_0, self.nx))
        # self.ocp_mhe.cost.Vx_0[:self.nx,:] = np.eye(self.nx)
        # self.ocp_mhe.cost.Vx_0[self.nx:2*self.nx,:] = np.eye(self.nx)
        #
        # self.ocp_mhe.cost.Vu_0 = np.zeros((self.ny_0, self.nu))
        # self.ocp_mhe.cost.Vu_0[self.nx:self.nx+self.nu,:] = np.eye(self.nu)

        self.ocp_mhe.cost.W_0 = scipy.linalg.block_diag(R, Q, Q0)
        self.ocp_mhe.model.cost_y_expr_0 = cs.vertcat(self.x[:self.nx],
                                                      self.u,
                                                      self.x)
        self.ocp_mhe.cost.yref_0 = np.zeros((self.ny_0,))

        # 2. Intermidiate stage (Lagrange term)
        # self.ocp_mhe.cost.Vx = np.zeros((self.ny, self.nx))
        # self.ocp_mhe.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        # self.ocp_mhe.cost.Vu = np.zeros((self.ny, self.nu))
        # self.ocp_mhe.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu)

        # 4. Weight for state cost
        self.ocp_mhe.cost.W = scipy.linalg.block_diag(R, Q)
        self.ocp_mhe.model.cost_y_expr = cs.vertcat(self.x[:self.nx],
                                                    self.u)
        self.ocp_mhe.cost.yref = np.zeros((self.ny,))

    def set_ocp_solver(self):

        # Set QP solver
        self.ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp_mhe.solver_options.integrator_type = 'ERK'

        self.ocp_mhe.solver_options.nlp_solver_type = 'SQP'
        self.ocp_mhe.solver_options.nlp_solver_max_iter = 200

        # Set prediction horizon
        self.ocp_mhe.solver_options.tf = self.tf

        self.acados_mhe_solver = AcadosOcpSolver(self.ocp_mhe)

    def get_ocp_solver(self):

        return self.acados_mhe_solver