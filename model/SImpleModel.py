from acados_template import AcadosModel
import casadi as cs

class SImpleModel:
    def __init__(self):
        '''
        Constructor method for SImpleModel
        '''

        self.model_name = "Simple_Model_For_MHE"

        self.model = AcadosModel()

        self.m = 2;

        self.p = cs.MX.sym('p',1)
        self.v = cs.MX.sym('v',1)
        self.x = cs.vertcat(self.p, self.v)

        self.dpdt = cs.MX.sym('dpdt',1)
        self.dvdt = cs.MX.sym('dvdt',1)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt)

        