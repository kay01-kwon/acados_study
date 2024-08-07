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
        self.p = cs.MX.sym('p', 1)
        self.v = cs.MX.sym('v', 1)
        self.m = cs.MX.sym('m', 1)
        self.x = cs.vertcat(self.p, self.v, self.m)
        self.x_dim = 3      # position, velocity, and mass

        self.w_p = cs.MX.sym('w_p', 1)
        self.w_v = cs.MX.sym('w_v', 1)
        self.w_m = cs.MX.sym('w_m', 1)
        self.w = cs.vertcat(self.w_p, self.w_v, self.w_m)

        # Control input declartaion (Pararmeter to pass)
        self.F = cs.MX.sym('F')

        # dxdt declaration
        self.dpdt = cs.MX.sym('dpdt',1)
        self.dvdt = cs.MX.sym('dvdt',1)
        self.dmdt = cs.MX.sym('dmdt',1)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt, self.dmdt)


    def get_acados_model(self):

        self.f_expl = cs.vertcat(self.p_kinematics(), self.v_dynamics(), 0)
        self.f_impl = cs.vertcat(self.dpdt, self.dvdt, self.dmdt)

        # Add state noise
        self.model.f_expl = self.f_expl + self.w
        self.model.f_impl = self.f_impl

        self.model.x = self.x
        self.model.u = self.w
        self.model.xdot = self.xdot
        self.model.p = self.param

        self.model.model_name = self.model_name

        return self.model

    def p_kinematics(self):

        return self.v

    def v_dynamics(self):

        return self.F/self.m

