# acados_study

## How to execute the program

```angular2html
python3 example_ocp_2d
```

When executing the command above, it generates json file and 
show the figure below.

<img src="figures/2d_mpc_result.png">

## MPC-ECBF

### ECBF formulation 
The vector $\eta_{b}(x)$ can be defined as 
$\eta_{b}(x) = 
\begin{bmatrix}
h(x)\\
\dot{h}(x)
\end{bmatrix}$

where $h(x) = (x-x_{obs})^2 + (y-y_{obs})^2$ and

$\dot{h}(x) = 2x(x-x_{obs})+2y(y-y_{obs}).$

The definition of $\mu$ is the following:

$L_{f}^2h(x) + L_{g}L_{f}h(x) =\mu.$

It can be written as 

$\begin{equation}
\begin{split}
\mu &= \frac{d^2h(x)}{dt^2}\\
    &=2(\dot{x}^2 + \dot{y}^2)
    + 2(x-x_{obs})\ddot{x}
    + 2(y-y_{obs})\ddot{y}.
\end{split}
\end{equation}$

Herein, $\ddot{x}$ and $\ddot{y}$ can be represented as
the force component like
$\ddot{x} = \frac{f_{x}}{m}$ and 
$\ddot{y} = \frac{f_{y}}{m}$, respectively.

<img src="figures/2d_mpc_w_ECBF_result1.png">

<img src="figures/2d_mpc_w_ECBF_result2.png">

<img src="figures/2d_mpc_w_ECBF_result3.png">