from acados_template import AcadosModel
import casadi as cs

class MheSimpleModel:
    def __init__(self):
        '''
        Simple model
        u = m * dxdt
        Guess m!
        '''
        self.model_name = "MheSimpleModel"

        self.model = AcadosModel()

        # State variables declaration
        self.p = cs.MX.sym('p',1)
        self.v = cs.MX.sym('v',1)
        self.m = cs.MX.sym('m')
        self.x = cs.vertcat(self.p, self.v, self.m)
        self.x_dim = 3      # position, velocity, and mass


        # Control input declartaion
        self.u = cs.MX.sym('u')

        # dxdt declaration
        self.dpdt = cs.MX.sym('dpdt',1)
        self.dvdt = cs.MX.sym('dvdt',1)
        self.dmdt = cs.MX.sym('dmdt',1)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt, self.dmdt)


    def get_acados_model(self):

        self.f_expl = cs.vertcat(self.p_kinematics(), self.v_dynamcis(), 0)
        self.f_impl = cs.vertcat(self.dpdt, self.dvdt, self.dmdt)

        self.model.f_expl = self.f_expl
        self.model.f_impl = self.f_impl

        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u

        self.model.model_name = self.model_name

        return self.model

    def p_kinematics(self):

        return self.v

    def v_dynamics(self):

        return self.u/self.m

