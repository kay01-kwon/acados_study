from acados_template import AcadosModel
import casadi as cs

class DisturbanceEstimator:
    def __init__(self, Parameter):
        '''
        Simple model
        u = m * dxdt + f
        Guess f!
        '''
        self.model_name = "disturbance_estimator"

        self.acados_model = AcadosModel()

        self.m_nom = Parameter['m_nom']

        # State variables declaration
        self.p = cs.MX.sym('p', 1)      # position
        self.v = cs.MX.sym('v', 1)      # velocity
        self.f = cs.MX.sym('f', 1)      # disturbance
        self.x = cs.vertcat(self.p, self.v, self.f)

        # Noise information
        self.w_p = cs.MX.sym('w_p', 1)
        self.w_v = cs.MX.sym('w_v', 1)
        self.w_f = cs.MX.sym('w_d', 1)
        self.w = cs.vertcat(self.w_p, self.w_v, self.w_f)

        # Control input declartaion (Pararmeter to pass)
        self.F = cs.MX.sym('F')

        # dxdt declaration
        self.dpdt = cs.MX.sym('dpdt',1)
        self.dvdt = cs.MX.sym('dvdt',1)
        self.dfdt = cs.MX.sym('dfdt',1)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt, self.dfdt)


    def get_acados_model(self):

        self.f_expl = cs.vertcat(self.p_kinematics(), self.v_dynamics(), 0)
        self.f_expl = self.f_expl + self.w
        self.f_impl = self.xdot

        # Add state noise
        self.acados_model.f_expl_expr = self.f_expl
        self.acados_model.f_impl_expr = self.f_impl - self.f_expl

        self.acados_model.x = self.x
        self.acados_model.u = self.w
        self.acados_model.xdot = self.xdot
        self.acados_model.p = self.F

        self.acados_model.name = self.model_name

        return self.acados_model

    def p_kinematics(self):

        return self.v

    def v_dynamics(self):

        return 1/self.m_nom*(self.F + self.f)

