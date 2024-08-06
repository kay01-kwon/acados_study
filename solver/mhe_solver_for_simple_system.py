import numpy as np
from acados_template import AcadosOcpSolver, AcadosOcp
from model.mhe_simple_model import MheSimpleModel
import casadi as cs

class MheSolverForSimpleSystem:
    def __init__(self, model, N, h, Q, Q0, R):
        '''

        :param model:
        :param N: Number of shooting nodes
        :param h: prediction horizon
        :param Q: Weight matrix for 
        :param Q0:
        :param R:
        '''

        self.ocp_mhe = AcadosOcp()

        self.ocp_mhe.model = MheSimpleModel().get_acados_model()

        self.nx_augmented = self.ocp_mhe.model.x.rows()
        self.nparam = self.ocp_mhe.model.p.rows()
        self.nx = self.nx_augmented-1

        self.ocp_mhe.dims.N = N

