# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2025/12/22 19:56
@version: 1.0
@File: ex_time_accuracy.py
'''
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import Data as pde_datas
from Solver_For_CHNS_Surfactant import *
from firedrake import *
from Tools.UtilityFunctions import PrintConvergenceTable, CallTimeConvergence

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pde_names = [
        "pde_CHNS_sur_data",
        # "pde_CHNS_sur_data_energy",
    ]
    pdeparas = {
        "M_phi": 5e-4,
        "M_rho": 5e-4,
        "S_phi": 0,
        "S_rho": 0,
        "gamma_1": 1,
        "gamma_2": 1,
        "alpha_1": 2,
        "alpha_2": 2,
        "kappa_lambda": 1e5,
        "kappa_zeta": 1e5,
        "beta": 0.01,
        "xi": 1e-3,
        "epsilon": 0.12,
        "eta": 0.08,
        "theta": 0.005,
        "sigma": 1e-4,
        "Re": 1
    }
    solvers = [
        #Solver_CHNS_Surfactant_1st,
        Solver_CHNS_Surfactant_2nd
    ]
    start_T = 0
    end_T = 1
    poly_order = 1
    options = {
        "energyflag": False,
        "return_errors": True
    }
    num = 5
    dt_list = [(1 / 4) * 1 / (2 ** t) for t in range(num)]

    for solver in solvers:
        # 动态获取pde类
        for name in pde_names:
            pde_class = getattr(pde_datas, name)
            if name in ["pde_CHNS_sur_data","pde_CHNS_sur_data_energy"]:
                domain = (0, 2*pi, 0, 4*pi) # (left, right, down, up)
                Nx, Ny = 100, 200
                if solver.__name__ == "Solver_CHNS_Surfactant_1st":
                    options["savename"] = name + "_1st"
                elif solver.__name__ == "Solver_CHNS_Surfactant_2nd":
                    options["savename"] = name + "_2nd"

            # create mesh
            mesh = PeriodicRectangleMesh(Nx, Ny, domain[1] - domain[0],
                                         domain[3] - domain[2],
                                         direction="both")
            coords = mesh.coordinates.dat.data[:]
            xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
            ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
            xcenter = 0.5 * (xmin + xmax)
            ycenter = 0.5 * (ymin + ymax)
            coords[:, 0] -= xcenter
            coords[:, 1] -= ycenter
            mesh.coordinates.dat.data[:] = coords

            # function space
            U = VectorFunctionSpace(mesh, "CG", poly_order + 1)
            W = FunctionSpace(mesh, "CG", poly_order + 1)
            V = FunctionSpace(mesh, "CG", poly_order)
            up_func_space = U * V
            phimu_func_space = W * W
            rhoomega_func_space = W * W
            ns_nullspace = ns_nullspace = MixedVectorSpaceBasis(up_func_space, [up_func_space.sub(0), VectorSpaceBasis(constant=True, comm=V.comm)])

            # build pde instant
            pde = pde_class(mesh, pdeparas)

            # prepare for solver kargs
            time_setting = (start_T, end_T, None)
            karg = {
                "time_setting": time_setting,
                "pde": pde,
                "up_func_space": up_func_space,
                "phimu_func_space": phimu_func_space,
                "rhoomega_func_space": rhoomega_func_space,
                "ns_nullspace": ns_nullspace,
                "options": options
            }

            if rank == 0:
                print(f"Now using {pde.name}")
                print(f"Nx = {Nx}, Ny = {Ny}")
                print(f"T = [{start_T}, {end_T}], dt = {dt_list}")
                print("Time Convergence Test:")
            data = CallTimeConvergence(solver, dt_list, karg)

            if rank == 0:
                headers = [
                    "dt", "phi_L2_error", "rate", "mu_L2_error", "rate",
                    "rho_L2_error", "rate", "omega_L2_error", "rate",
                    "u_L2_error", "rate", "p_L2_error", "rate",
                    "lambda_L2_error", "rate", "zeta_L2_error", "rate"
                ]
                PrintConvergenceTable(data, headers)

"""
bug: 
    rho_now, omega_now = split(phimu_now) # 写串了
    rho_next, omega_next = split(phimu_next) # 写串了
    A_lambda 漏乘 rho_next_2
    func_dlambda_int_F_non 系数问题
    
Now using pde_CHNS_Sur_data
Nx = 100, Ny = 200
T = [0, 1], dt = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
Time Convergence Test:
|   dt   | phi_L2_error |  rate  | mu_L2_error |  rate  | rho_L2_error |  rate  | omega_L2_error |  rate  | u_L2_error |  rate  | p_L2_error |  rate  | lambda_L2_error |  rate  | zeta_L2_error |  rate  |
| 0.5000 |  1.2527e+00  | 0.0000 | 3.2094e+01  | 0.0000 |  9.0172e-01  | 0.0000 |   7.0996e+02   | 0.0000 | 2.8177e+00 | 0.0000 | 1.8433e+01 | 0.0000 |   8.3104e-03    | 0.0000 |  6.3616e-05   | 0.0000 |
| 0.2500 |  6.0423e-01  | 1.0518 | 1.9938e+01  | 0.6868 |  4.6089e-01  | 0.9683 |   3.6320e+02   | 0.9670 | 9.6147e-01 | 1.5512 | 9.7864e+00 | 0.9134 |   3.1996e-03    | 1.3770 |  3.3355e-05   | 0.9315 |
| 0.1250 |  3.0344e-01  | 0.9937 | 1.1075e+01  | 0.8482 |  2.3303e-01  | 0.9839 |   1.8432e+02   | 0.9786 | 3.3847e-01 | 1.5062 | 5.1168e+00 | 0.9355 |   1.3776e-03    | 1.2157 |  1.1261e-05   | 1.5665 |
| 0.0625 |  1.5318e-01  | 0.9862 | 5.8784e+00  | 0.9138 |  1.1717e-01  | 0.9919 |   9.2937e+01   | 0.9879 | 1.2729e-01 | 1.4110 | 2.6380e+00 | 0.9558 |   6.3556e-04    | 1.1161 |  3.8085e-06   | 1.5641 |
| 0.0312 |  7.7150e-02  | 0.9895 | 3.0423e+00  | 0.9503 |  5.8749e-02  | 0.9959 |   4.6677e+01   | 0.9935 | 5.1876e-02 | 1.2949 | 1.3442e+00 | 0.9726 |   3.0477e-04    | 1.0603 |  1.3929e-06   | 1.4511 |
| 0.0156 |  3.8745e-02  | 0.9936 | 1.5502e+00  | 0.9727 |  2.9416e-02  | 0.9980 |   2.3393e+01   | 0.9967 | 2.2787e-02 | 1.1869 | 6.7936e-01 | 0.9845 |   1.4917e-04    | 1.0308 |  5.6115e-07   | 1.3116 |

Now using pde_CHNS_Sur_data
Nx = 100, Ny = 200
T = [0, 1], dt = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
Time Convergence Test:
|   dt   | phi_L2_error |  rate   | mu_L2_error |  rate   | rho_L2_error |  rate   | omega_L2_error |  rate   | u_L2_error |  rate   | p_L2_error |  rate   | lambda_L2_error |   rate   | zeta_L2_error |   rate   |
| 0.5000 |  8.4681e-01  | 0.0000  | 3.7996e+01  | 0.0000  |  6.2907e-01  | 0.0000  |   9.0470e+01   | 0.0000  | 6.4302e-01 | 0.0000  | 2.2565e+01 | 0.0000  |   3.0940e-03    |  0.0000  |  3.0438e-05   |  0.0000  |
| 0.2500 |  2.2886e-01  | 1.8875  | 5.7646e+00  | 2.7206  |  1.7487e-01  | 1.8470  |   1.2364e+01   | 2.8713  | 9.4448e-02 | 2.7673  | 4.6993e+00 | 2.2636  |   4.0321e-04    |  2.9398  |  6.5399e-06   |  2.2185  |
| 0.1250 |  5.7152e-02  | 2.0016  | 1.4809e+00  | 1.9607  |  4.3313e-02  | 2.0134  |   1.5182e+00   | 3.0257  | 1.2717e-02 | 2.8927  | 1.1661e+00 | 2.0108  |   4.9481e-05    |  3.0266  |  1.3349e-06   |  2.2926  |
| 0.0625 |  1.4312e-02  | 1.9976  | 3.8040e-01  | 1.9609  |  1.0648e-02  | 2.0241  |   3.7510e-01   | 2.0170  | 2.3035e-03 | 2.4649  | 2.9082e-01 | 2.0035  |   3.8018e-06    |  3.7021  |  2.9356e-07   |  2.1850  |
| 0.0312 |  3.5693e-03  | 2.0035  | 9.5729e-02  | 1.9905  |  2.6371e-03  | 2.0136  |   1.4783e-01   | 1.3434  | 5.2021e-04 | 2.1466  | 7.2712e-02 | 1.9999  |   3.3247e-07    |  3.5154  |  7.0166e-08   |  2.0648  |
| 0.0156 |  8.9002e-04  | 2.0037  | 2.3961e-02  | 1.9983  |  6.6961e-04  | 1.9775  |   8.8471e-02   | 0.7407  | 1.2694e-04 | 2.0350  | 2.4902e-02 | 1.5459  |   2.5942e-07    |  0.3580  |  1.7261e-08   |  2.0233  |
"""