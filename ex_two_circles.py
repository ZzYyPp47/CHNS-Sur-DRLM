# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2025/12/22 19:56
@version: 1.0
@File: ex_two_circles.py
'''
import Data as pde_datas
from Solver_For_CHNS_Surfactant import *
from firedrake import *
from Tools.UtilityFunctions import PrintConvergenceTable, CallTimeConvergence

if __name__ == '__main__':
    pde_names = [
        "pde_CHNS_sur_twocircles_1",
        "pde_CHNS_sur_twocircles_2"
    ]
    solvers = [
        Solver_CHNS_Surfactant_1st
    ]
    poly_order = 1
    dt_list = [0.001]

    for solver in solvers:
        # 动态获取pde类
        for name in pde_names:
            pde_class = getattr(pde_datas, name)
            domain = (0, 2 * pi, 0, 2 * pi)  # (left, right, down, up)
            Nx, Ny = 128, 128
            options = {
                "savename": name,
                "vtksave": False,
                "return_errors": False,
                "debug": False,
                "energyflag": True
            }
            if name in ["pde_CHNS_sur_twocircles_1"]:
                start_T = 0
                end_T = 2
                pdeparas = {
                    "M_phi": 1e-3,
                    "M_rho": 1e-3,
                    "S_phi": 10,
                    "S_rho": 10,
                    "gamma_1": 1,
                    "gamma_2": 1,
                    "alpha_1": 0,
                    "alpha_2": 0,
                    "kappa_lambda": 1e10,
                    "kappa_zeta": 1e6,
                    "beta": 0,
                    "xi": 1e-3,
                    "epsilon": 0.04,
                    "eta": 0.01,
                    "theta": 3,
                    "sigma": 1e-3,
                    "Re": 1
                }
            elif name in ["pde_CHNS_sur_twocircles_2"]:
                start_T = 0
                end_T = 2
                pdeparas = {
                    "M_phi": 1e-3,
                    "M_rho": 1e-3,
                    "S_phi": 10,
                    "S_rho": 10,
                    "gamma_1": 1,
                    "gamma_2": 1,
                    "alpha_1": 0,
                    "alpha_2": 0,
                    "kappa_lambda": 1e10,
                    "kappa_zeta": 1e6,
                    "beta": 0,
                    "xi": 1e-3,
                    "epsilon": 0.04,
                    "eta": 0.01,
                    "theta": 0.01,
                    "sigma": 1e-3,
                    "Re": 1
                }
            elif name in ["pde_CHNS_sur_twocircles_3"]:
                start_T = 0
                end_T = 4
                pdeparas = {
                    "M_phi": 1e-2,
                    "M_rho": 1e-2,
                    "S_phi": 10,
                    "S_rho": 10,
                    "gamma_1": 1,
                    "gamma_2": 1,
                    "alpha_1": 2,
                    "alpha_2": 2,
                    "kappa_lambda": 1e10,
                    "kappa_zeta": 1e6,
                    "beta": 4,
                    "xi": 1e-3,
                    "epsilon": 0.04,
                    "eta": 0.01,
                    "theta": 2,
                    "sigma": 1e-3,
                    "Re": 1
                }

            # create mesh
            mesh = PeriodicRectangleMesh(Nx, Ny, domain[1],
                                         domain[3],
                                         direction="both")

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
            print(f"Now using {pde.name}")
            print(f"Nx = {Nx}, Ny = {Ny}")
            print(f"T = [{start_T}, {end_T}], dt = {dt_list}")
            CallTimeConvergence(solver, dt_list, karg)