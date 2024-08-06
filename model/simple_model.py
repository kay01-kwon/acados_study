from acados_template import AcadosModel
import casadi as cs

class SimpleModel:
    def __init__(self, mass = 2):
        '''
        Constructor method for SImpleModel
        '''

        self.model_name = "Simple_Model_For_MHE"

        self.model = AcadosModel()

        self.m = mass

        self.p = cs.MX.sym('p',1)
        self.v = cs.MX.sym('v',1)
        self.x = cs.vertcat(self.p, self.v)

        self.u = cs.MX.sym('u')

        self.dpdt = cs.MX.sym('dpdt',1)
        self.dvdt = cs.MX.sym('dvdt',1)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt)


    def get_acados_model(self):

        self.f_expl = cs.vertcat(self.p_kinematics(), self.v_dynamics)
        self.f_impl = cs.vertcat(self.dpdt, self.dvdt)

        self.model.f_expl_expr = self.f_expl
        self.model.f_impl_expr = self.f_impl

        self.model.x = self.x
        self.model.u = self.u
        self.model.xdot = self.xdot

        self.model.model_name = self.model_name

        return self.model


    def p_kinematics(self):

        return self.v

    def v_dynamics(self):

        return self.u/self.m