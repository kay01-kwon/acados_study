import numpy as np
import scipy.linalg
from acados_template import AcadosOcpSolver, AcadosOcp
from model.mhe_simple_model import MheSimpleModel
import casadi as cs
from scipy.linalg
class MheSolverForSimpleSystem:
    def __init__(self, N, Tf, Q, Q0, R):
        '''
        :param N: Number of shooting nodes
        :param Tf: prediction horizon
        :param Q: Weight matrix for the observed state variables
        :param Q0: Terminal weight matrix for state variables
        :param R: Weight matrix for the noise of the observed state variables
        '''

        self.ocp_mhe = AcadosOcp()

        self.ocp_mhe.model = MheSimpleModel().get_acados_model()

        # Dimension info
        self.nx_augmented = self.ocp_mhe.model.x.rows()
        self.nparam = self.ocp_mhe.model.p.rows()
        self.nx = self.nx_augmented-1
        self.ny = Q.shape[0] + R.shape[0]                   # state, noise
        self.nu = self.ocp_mhe.model.u.rows()
        self.ny_e = 0
        self.ny_0 = Q.shape[0] + R.shape[0] + Q0.shape[0]   # state, noise, Arrival cost

        # Set the number of shooting nodes

        self.ocp_mhe.dims.N = N

        self.ocp_mhe.cost.cost_type = 'LINEAR_LS'
        self.ocp_mhe.cost.cost_type_e = 'LINEAR_LS'

        # Setup weight for cost

        # 1. Vx weight
        self.ocp_mhe.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp_mhe.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

        # 2. Vx arrival weight
        self.ocp_mhe.cost.Vx_e = np.zeros((self.nx,))
        self.ocp_mhe.cost.Vx_e = np.eye(self.nx,)

        # 3. Vu weight
        self.ocp_mhe.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp_mhe.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu)

        # 4. Weight for state cost
        self.ocp_mhe.cost.W = scipy.linalg.block_diag(Q, R)

        







