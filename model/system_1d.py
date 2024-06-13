from acados_template import AcadosModel
from casadi import SX, vertcat

def export_dynamics_ode_model() -> AcadosModel:

    model_name = 'simple_system'

    # dxdt = v
    # dvdt = 1/m * u

    # Mass property
    m = 1.

    # Set up states and control
    x1 = SX.sym('x1')
    v1 = SX.sym('v1')
    x = vertcat(x1, v1)

    u = SX.sym('u')

    # xdot = d/dt[x; v]

    dxdt = SX.sym('dxdt')
    dvdt = SX.sym('dvdt')

    xdot = vertcat(dxdt, dvdt)

    f_expl = vertcat(v1,
                     1/m*u)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$ [m]","$v$ [m/s]"]
    model.u_label = "$u$ [N]"

    return model