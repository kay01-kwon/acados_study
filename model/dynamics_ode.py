from acados_template import AcadosModel
from casadi import SX, vertcat

def export_dynamics_ode_model() -> AcadosModel:

    model_name = '1d_double_intergral'

    # dxdt = v
    # dvdt = 1/m * u

    # Mass property
    m = 1.

    # Set up states and control
    x = SX.sym('x')
    v = SX.sym('v')

    u = SX.sym('u')

    # xdot = d/dt[x; v]

    dxdt = SX.sym('dxdt')
    dvdt = SX.sym('dvdt')

    xdot = vertcat(dxdt, dvdt)

    f_expl = vertcat(v,
                     1/m*u)

    f_impl = dsdt - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    

    return model