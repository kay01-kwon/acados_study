from acados_template import AcadosModel
import casadi as cs

def export_dynamics_ode_model() -> AcadosModel:

    model_name = 'system_2d'

    # dpdt = v
    # dvdt = 1/m * u

    # Mass property
    m = 1.

    # Set up states and control
    p = cs.MX.sym('p',2)
    v = cs.MX.sym('v',2)
    x = cs.vertcat(p, v)

    u_x = cs.MX.sym('ux')
    u_y = cs.MX.sym('uy')
    u = cs.vertcat(u_x, u_y)

    # xdot = d/dt[x; v]

    dxdt = cs.MX.sym('dxdt',2)
    dvdt = cs.MX.sym('dvdt',2)

    xdot = cs.vertcat(dxdt, dvdt)

    f_expl = cs.vertcat(v,
                        1/m*u_x,
                        1/m*u_y)

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