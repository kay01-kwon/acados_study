import numpy as np
import scipy.linalg
from acados_template import AcadosOcpSolver, AcadosOcp
from model.mhe_simple_model import MheSimpleModel
import casadi as cs

class MheSolverForSimpleSystem:
    def __init__(self, N, Tf, Q, R, R0):
        '''
        :param N: Number of shooting nodes
        :param Tf: prediction horizon
        :param Q: Weight matrix for the observed state variables
        :param Q0: Terminal weight matrix for state variables
        :param R: Weight matrix for the noise of the observed state variables
        '''

        self.ocp_mhe = AcadosOcp()

        self.ocp_mhe.model = MheSimpleModel().get_acados_model()

        self.Tf = Tf

        # Dimension info
        self.nx_augmented = self.ocp_mhe.model.x.rows()
        self.nparam = self.ocp_mhe.model.p.rows()
        self.nx = self.nx_augmented-1
        self.ny = Q.shape[0] + R.shape[0]                   # state, noise
        self.nu = self.ocp_mhe.model.u.rows()
        self.ny_e = 0
        self.ny_0 = Q.shape[0] + R.shape[0] + R0.shape[0]   # state, noise, Arrival cost

        # Set the number of shooting nodes

        self.ocp_mhe.dims.N = N

        self.ocp_mhe.cost.cost_type = 'LINEAR_LS'       # Linear least square
        self.ocp_mhe.cost.cost_type_e = 'LINEAR_LS'     # Linear least square
        self.ocp_mhe.cost.cost_type_0 = 'LINEAR_LS'     # Linear least square

        self.ocp_mhe.parameter_values = np.zeros((self.nparam,))

        self.set_ocp_cost()

        self.set_ocp_solver()

        self.acados_mhe_solver = AcadosOcpSolver(self.ocp_mhe)

        self.acados_mhe_solver.cost_set(0, "W", scipy.linalg.block_diag(Q, R, R0))


        def set_ocp_cost(self):
            # Setup weight for cost

            # 1. Vx weight (Lagrangian term)
            self.ocp_mhe.cost.Vx = np.zeros((self.ny, self.nx))
            self.ocp_mhe.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

            # 2. Vx (Mayer term)
            self.ocp_mhe.cost.Vx_e = np.zeros((self.nx,))
            self.ocp_mhe.cost.Vx_e = np.eye(self.nx, )

            # 3. Vu weight
            self.ocp_mhe.cost.Vu = np.zeros((self.ny, self.nu))
            self.ocp_mhe.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu)

            # 4. Weight for state cost
            self.ocp_mhe.cost.W_0 = scipy.linalg.block_diag(Q, R, R0)
            self.ocp_mhe.cost.W = scipy.linalg.block_diag(Q, R)

            self.ocp_mhe.cost.yref = np.zeros((self.ny,))
            self.ocp_mhe.cost.yref_e = np.zeros((self.ny_e,))
            self.ocp_mhe.cost.yref_0 = np.zeros((self.ny_0,))

        def set_ocp_solver(self):

            # Set QP solver
            self.ocp_mhe.solver.options.qp_solver = 'FULL_CONDENSING_QPOASES'
            self.ocp_mhe.solver.options.hessian_approx = 'GAUSS_NEWTON'
            self.ocp_mhe.solver.options.integrator_type = 'ERK'

            self.ocp_mhe.solver.options.nlp_solver_type = 'SQP'
            self.ocp_mhe.solver.options.nlp_solver_max_iter = 200

            # Set prediction horizon
            self.ocp_mhe.solver.options.tf = self.Tf

        def get_ocp_solver(self):

            return self.acados_mhe_solver












