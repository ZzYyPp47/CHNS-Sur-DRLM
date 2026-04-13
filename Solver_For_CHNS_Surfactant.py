# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2025/12/22 19:56
@version: 1.0
@File: Solver_For_CHNS_Surfactant.py
'''

import time
import numpy as np
import os
from firedrake import *
from math import ceil
from tqdm import trange
from mpi4py import MPI
from Tools.UtilityFunctions import PrintProgressBar

# 迭代法求解，并行效率高且内存友善，但是迭代次数和容许误差需仔细选择
PHI_SOLVER_PARAMS = {
    "ksp_type": "gmres",
    "pc_type": "bjacobi",
    "sub_ksp_type": "preonly",
    "sub_pc_type": "ilu",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 500,
    # 'snes_monitor': None,
    # 'snes_view': None,
    # 'ksp_monitor_true_residual': None,
    # "log_view": None,
    # 'snes_converged_reason': None,
    # 'ksp_converged_reason': None,
}
NS_SOLVER_PARAMS = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "FULL",
    "pc_fieldsplit_schur_precondition": "selfp",  # 使用压力矩阵近似 Schur 补
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_0_pc_hypre_type": "boomeramg",
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "hypre",  # 压力块用 hypre
    "fieldsplit_1_pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 200,
    "ksp_gmres_restart": 100,
    # 'snes_monitor': None,
    # 'snes_view': None,
    # 'ksp_monitor_true_residual': None,
    # "log_view": None,
    # 'snes_converged_reason': None,
    # 'ksp_converged_reason': None,
}

# 最稳定且鲁棒，LU求解，缺点是内存大
PHI_SOLVER_PARAMS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
NS_SOLVER_PARAMS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

# # 关闭, 默认也是使用LU
# PHI_SOLVER_PARAMS, NS_SOLVER_PARAMS = None, None

def cal_energy(pde, phi, rho, u):
    return 0.5 * norm(u) ** 2 + float(pde.gamma_1) / 2 * norm(grad(phi)) ** 2 + float(pde.gamma_2) / 2 * norm(grad(rho)) ** 2 + assemble(func_F_non(pde, phi, rho) * dx)

def func(pde, phi):
    return (phi ** 3 - phi)

def Func(pde, phi):
    return (1 / 4) * (phi ** 2 - 1) ** 2

def func_h(pde, phi, rho):
    return func(pde, phi) / pde.epsilon ** 2 + pde.theta * div(rho * grad(phi)) - pde.xi * div(dot(grad(phi), grad(phi)) * grad(phi)) + pde.alpha_1 * rho * phi - pde.alpha_2 * rho

def func_G(pde, rho):
    G1 = rho * ln(rho) + (1 - rho) * ln(pde.sigma) + (1 - rho) ** 2 / (2 * pde.sigma) - pde.sigma / 2
    G2 = rho * ln(rho) + (1 - rho) * ln(1 - rho)
    G3 = (1 - rho) * ln(1 - rho) + rho * ln(pde.sigma) + rho ** 2 / (2 * pde.sigma) - pde.sigma / 2
    return conditional(
        ge(rho, 1 - pde.sigma),
        G1,
        conditional(
            le(rho, pde.sigma),
            G3,
            G2
        )
    )

def func_g(pde, rho):
    g1 = ln(rho / pde.sigma) + 1 - (1 - rho) / pde.sigma
    g2 = ln(rho / (1 - rho))
    g3 = ln(pde.sigma / (1 - rho)) - 1 + rho / pde.sigma
    return conditional(
        ge(rho, 1 - pde.sigma),
        g1,
        conditional(
            le(rho, pde.sigma),
            g3,
            g2
        )
    )

def func_q(pde, phi, rho):
    return func_g(pde, rho) / pde.eta ** 2 - pde.theta / 2 * dot(grad(phi), grad(phi)) + pde.alpha_1 / 2 * phi ** 2 - pde.alpha_2 * phi - pde.beta * rho

def func_F_non(pde, phi, rho):
    return Func(pde, phi) / pde.epsilon ** 2 + func_G(pde, rho) / pde.eta ** 2 - pde.theta / 2 * rho * dot(grad(phi),grad(phi)) + pde.xi / 4 * dot(grad(phi),grad(phi)) ** 2 + pde.alpha_1 / 2 * rho * phi ** 2 - pde.alpha_2 * rho * phi - pde.beta / 2 * rho ** 2

def func_dlambda_int_F_non(pde, phi_1, phi_2, rho_1, rho_2, _lambda):
    phi = phi_1 + _lambda * phi_2
    rho = rho_1 + _lambda * rho_2
    h_val = func_h(pde, phi, rho)  # δF_non/δφ
    q_val = func_q(pde, phi, rho)  # δF_non/δρ
    return assemble((h_val * phi_2 + q_val * rho_2) * dx)

def newton_armijo(func, d_func, init_guess=0, tol=1e-8, max_iter=300, c=1e-4, rho=0.5):
    x = init_guess
    for ii in range(max_iter):
        f_val = func(x)
        if abs(f_val) < tol:
            break
        df = d_func(x)
        if df == 0:
            break
        d = -f_val / df
        alpha = 1.0
        # 回溯线搜索
        for jj in range(20):  # 最多尝试20次缩小
            x_new = x + alpha * d
            if abs(func(x_new)) <= (1 - c * alpha) * abs(f_val):
                break
            alpha *= rho
        x = x + alpha * d
        # print(f"iteration {ii+1}: guess = {x}, search iter = {jj + 1}.")
    return x

def solve_quadratic(A,B,C):
    x_1 = float((-B + sqrt(B ** 2 - 4 * A * C)) / (2 * A))
    x_2 = float((-B - sqrt(B ** 2 - 4 * A * C)) / (2 * A))
    return x_1, x_2

def Solver_CHNS_Surfactant_1st(pde, time_setting, up_func_space, phimu_func_space, rhoomega_func_space, ns_nullspace, options):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.time()

    # unpacked the time settings
    start_T, end_T, dt = time_setting
    num_steps = ceil((end_T - start_T) / dt)
    pde.t.assign(start_T)

    # define phi_next_1 phi_next_2 phi_next phi_now mu_next_1 mu_next_2 mu_next mu_now
    phimu_next_1, phimu_next_2, phimu_now, phimu_next = Function(phimu_func_space), Function(phimu_func_space), Function(phimu_func_space), Function(phimu_func_space)
    phi_now, mu_now = split(phimu_now)
    phi_next, mu_next = split(phimu_next)
    phi_next_1, mu_next_1 = split(phimu_next_1)
    phi_next_2, mu_next_2 = split(phimu_next_2)

    # define rho_next_1 rho_next_2 rho_next rho_now omega_next_1 omega_next_2 omega_next omega_now
    rhoomega_next_1, rhoomega_next_2, rhoomega_now, rhoomega_next = Function(rhoomega_func_space), Function(rhoomega_func_space), Function(rhoomega_func_space), Function(rhoomega_func_space)
    rho_now, omega_now = split(rhoomega_now)
    rho_next, omega_next = split(rhoomega_next)
    rho_next_1, omega_next_1 = split(rhoomega_next_1)
    rho_next_2, omega_next_2 = split(rhoomega_next_2)

    # define up_next_1 up_next_2 up_next up_now
    up_next_1, up_next_2, up_now, up_next = Function(up_func_space), Function(up_func_space), Function(up_func_space), Function(up_func_space)
    u_now, p_now = split(up_now)
    u_next, p_next = split(up_next)
    u_next_1, p_next_1 = split(up_next_1)
    u_next_2, p_next_2 = split(up_next_2)

    # define eta_next eta_now q_next q_now
    lambda_now, lambda_next = Constant(1.0), Constant(1.0)
    zeta_now, zeta_next = Constant(1.0), Constant(1.0)

    # set initial conditions
    phimu_now.sub(0).interpolate(pde.init_phi)
    phimu_now.sub(1).interpolate(pde.init_mu)
    rhoomega_now.sub(0).interpolate(pde.init_rho)
    rhoomega_now.sub(1).interpolate(pde.init_omega)
    up_now.sub(0).interpolate(pde.init_u)
    up_now.sub(1).interpolate(pde.init_p)

    bc_u_up = DirichletBC(up_func_space.sub(0), pde.bc_u_up, 4) if hasattr(pde,"bc_u_up") else None
    bc_u_down = DirichletBC(up_func_space.sub(0), pde.bc_u_down, 3) if hasattr(pde, "bc_u_down") else None
    bc_u_left = DirichletBC(up_func_space.sub(0), pde.bc_u_left, 1) if hasattr(pde,"bc_u_left") else None
    bc_u_right = DirichletBC(up_func_space.sub(0), pde.bc_u_right, 2) if hasattr(pde,"bc_u_right") else None
    bc_u_all = DirichletBC(up_func_space.sub(0), pde.bc_u_all, "on_boundary") if hasattr(pde,"bc_u_all") else None
    bc_u0 = DirichletBC(up_func_space.sub(0), Constant([0] * up_func_space.mesh().geometric_dimension()), "on_boundary")

    # set trival and test functions
    phi_trial, mu_trial = TrialFunctions(phimu_func_space)
    phi_test, mu_test = TestFunctions(phimu_func_space)

    rho_trial, omega_trial = TrialFunctions(rhoomega_func_space)
    rho_test, omega_test = TestFunctions(rhoomega_func_space)
    
    u_trial, p_trial = TrialFunctions(up_func_space)
    u_test, p_test = TestFunctions(up_func_space)

    # step 1 (I): solve phi_next_1 phi_next_2 mu_next_1 mu_next_2
    equa_phi_L1 = (
        inner(phi_trial, phi_test) * dx
        + dt * pde.M_phi * inner(grad(mu_trial),grad(phi_test)) * dx
    )
    equa_phi_L2 = (
        pde.gamma_1 * inner(grad(phi_trial),grad(mu_test)) * dx
        + pde.S_phi / pde.epsilon ** 2 * inner(phi_trial,mu_test) * dx
        - inner(mu_trial,mu_test) * dx
    )

    if pde.rhs_phi is not None:
        equa_phi_R1 = (
            inner(phi_now, phi_test) * dx
            + dt * inner(pde.rhs_phi,phi_test) * dx
        )
    else:
        equa_phi_R1 = (
            inner(phi_now, phi_test) * dx
        )
    equa_phi_R2 = (
        pde.S_phi / pde.epsilon ** 2 * inner(phi_now,mu_test) * dx
    )
    equa_phi_R3 = (
        - dt * inner(div(u_now * phi_now), phi_test) * dx
    )
    equa_phi_R4 = (
        - inner(func_h(pde,phi_now, rho_now),mu_test) * dx
    )

    # step 1 (II): solve rho_next_1 rho_next_2 omega_next_1 omega_next_2
    equa_rho_L1 = (
        inner(rho_trial, rho_test) * dx
        + dt * pde.M_rho * inner(grad(omega_trial),grad(rho_test)) * dx
    )
    equa_rho_L2 = (
        pde.gamma_2 * inner(grad(rho_trial),grad(omega_test)) * dx
        + pde.S_rho / pde.eta ** 2 * inner(rho_trial,omega_test) * dx
        - inner(omega_trial,omega_test) * dx
    )

    if pde.rhs_rho is not None:
        equa_rho_R1 = (
            inner(rho_now, rho_test) * dx
            + dt * inner(pde.rhs_rho,rho_test) * dx
        )
    else:
        equa_rho_R1 = (
            inner(rho_now, rho_test) * dx
        )
    equa_rho_R2 = (
        pde.S_rho / pde.eta ** 2 * inner(rho_now,omega_test) * dx
    )
    equa_rho_R3 = (
        - dt * inner(div(u_now * rho_now), rho_test) * dx
    )
    equa_rho_R4 = (
        - inner(func_q(pde,phi_now, rho_now),omega_test) * dx
    )

    # step 2: solve up_next_1 up_next_2
    equa_ns_L = (
        inner(u_trial,u_test) * dx
        + (dt / pde.Re) * inner(grad(u_trial),grad(u_test)) * dx
        - dt * inner(p_trial, div(u_test)) * dx
        + inner(div(u_trial),p_test) * dx
    )

    if pde.rhs_ns is not None:
        equa_ns_R1 = (
            inner(u_now, u_test) * dx
            + dt * inner(pde.rhs_ns - lambda_next * (phi_next * grad(mu_next) + rho_next * grad(omega_next)), u_test) * dx
        )
    else:
        equa_ns_R1 = (
            inner(u_now, u_test) * dx
            - dt * inner(lambda_next * (phi_next * grad(mu_next) + rho_next * grad(omega_next)), u_test) * dx
        )
    equa_ns_R2 = (
        - dt * inner(dot(grad(u_now), u_now),u_test) * dx
    )

    # step solver
    if "vtksave" in options and options["vtksave"]:
        phi, mu = phimu_now.subfunctions
        rho, omega = rhoomega_now.subfunctions
        u, p = up_now.subfunctions
        phi.rename("phi")
        mu.rename("mu")
        rho.rename("rho")
        omega.rename("omega")
        u.rename("u")
        p.rename("p")
        outfile = VTKFile(f"data/{options["savename"]}.pvd")
        outfile.write(phi, mu, rho, omega, u, p)

    if "energyflag" in options and options["energyflag"]:
        os.makedirs("data", exist_ok=True)
        with open(f"data/{options["savename"]}.txt", 'w') as f:
            f.write(f"t,phi_mass,rho_mass,phi_diffnorm,rho_diffnorm,lambda^2,zeta^2,energy\n")
            f.write(f"{float(pde.t)},{assemble(inner(phi_now,1) * dx)},{assemble(inner(rho_now, 1) * dx)},nan,nan,{float(lambda_now) ** 2},{float(zeta_now) ** 2},{cal_energy(pde, phi_now, rho_now, u_now)}\n")

    phi_A = assemble(equa_phi_L1 + equa_phi_L2)
    rho_A = assemble(equa_rho_L1 + equa_rho_L2)
    nullspace_rho = None
    nullspace_phi = None
    phi_solver = LinearSolver(phi_A, solver_parameters=PHI_SOLVER_PARAMS, nullspace=nullspace_phi)
    rho_solver = LinearSolver(rho_A, solver_parameters=PHI_SOLVER_PARAMS, nullspace=nullspace_rho)

    bcs = [bc for bc in [bc_u_up, bc_u_down, bc_u_left, bc_u_right, bc_u_all] if bc is not None] or None

    for step in range(num_steps):
        next_t = start_T + (step + 1) * dt
        pde.t.assign(next_t)

        ##  solve step 1 (start):
        phi_b1 = assemble(equa_phi_R1 + equa_phi_R2)
        phi_solver.solve(phimu_next_1, phi_b1)

        phi_b2 = assemble(equa_phi_R3 + equa_phi_R4)
        phi_solver.solve(phimu_next_2, phi_b2)

        rho_b1 = assemble(equa_rho_R1 + equa_rho_R2)
        rho_solver.solve(rhoomega_next_1, rho_b1)

        rho_b2 = assemble(equa_rho_R3 + equa_rho_R4)
        rho_solver.solve(rhoomega_next_2, rho_b2)

        A_lambda = pde.kappa_lambda - assemble((func_h(pde, phi_now, rho_now) * phi_next_2 + func_q(pde, phi_now, rho_now) * rho_next_2) * dx)
        B_lambda = - assemble((func_h(pde, phi_now, rho_now) * (phi_next_1 - phi_now) + func_q(pde, phi_now, rho_now) * (rho_next_1 - rho_now)) * dx)
        C_lambda = - pde.kappa_lambda * lambda_now ** 2 - assemble(func_F_non(pde, phi_now, rho_now) * dx)
        _lambda = newton_armijo(
            func=lambda _lambda: float(Constant(assemble(func_F_non(pde, phi_next_1 + _lambda * phi_next_2, rho_next_1 + _lambda * rho_next_2) * dx) + A_lambda * _lambda ** 2 + B_lambda * _lambda + C_lambda)),
            d_func=lambda _lambda: float(Constant(func_dlambda_int_F_non(pde, phi_next_1, phi_next_2, rho_next_1, rho_next_2, _lambda) + 2 * A_lambda * _lambda + B_lambda)),
            init_guess=1
        )
        lambda_next.assign(_lambda)
        # compute phi_next mu_next:
        phimu_next.assign(phimu_next_1 + lambda_next * phimu_next_2)
        # compute rho_next omega_next:
        rhoomega_next.assign(rhoomega_next_1 + lambda_next * rhoomega_next_2)
        #  solve step 1 (end)

        if "debug" in options and options["debug"]:
            print(f"phi_now = {norm(phi_now)}, mu_now = {norm(mu_now)}, lambda_now = {float(lambda_now)}, lambda_next = {float(lambda_next)}") # debug
            print(f"exact_phi = {norm(pde.exact_phi)}, phi_next = {norm(phi_next)}, exact_mu = {norm(pde.exact_mu)}, mu_next = {norm(mu_next)}") # debug
            print(f"rho_now = {norm(rho_now)}, omega_now = {norm(omega_now)}, lambda_now = {float(lambda_now)}, lambda_next = {float(lambda_next)}") # debug
            print(f"exact_rho = {norm(pde.exact_rho)}, rho_next = {norm(rho_next)}, exact_omega = {norm(pde.exact_omega)}, omega_next = {norm(omega_next)}") # debug
        # phimu_next.sub(0).interpolate(pde.exact_phi) # debug
        # phimu_next.sub(1).interpolate(pde.exact_mu) # debug
        # rhoomega_next.sub(0).interpolate(pde.exact_rho) # debug
        # rhoomega_next.sub(1).interpolate(pde.exact_omega) # debug

        ##  solve step 2 (start):
        solve(equa_ns_L == equa_ns_R1, up_next_1, bcs=bcs, nullspace=ns_nullspace, solver_parameters=NS_SOLVER_PARAMS)
        solve(equa_ns_L == equa_ns_R2, up_next_2, bcs=bc_u0, nullspace=ns_nullspace, solver_parameters=NS_SOLVER_PARAMS)

        A_zeta = pde.kappa_zeta + 0.5 * assemble(inner(u_next_2, u_next_2) * dx) + dt * (1 / pde.Re) * assemble(inner(grad(u_next_2),grad(u_next_2)) * dx)
        B_zeta = - assemble(dot(u_next_1 - u_now, u_next_2) * dx) - dt * assemble((lambda_next * dot(phi_next * grad(mu_next) + rho_next * grad(omega_next),u_next_2) + dot(u_next_1,dot(grad(u_now), u_now))) * dx)
        C_zeta = - pde.kappa_zeta * zeta_now ** 2 - 0.5 * assemble(inner(u_next_1 - u_now, u_next_1 - u_now) * dx) - dt * lambda_next * assemble((div(u_now * phi_now) * mu_next + dot(u_next_1, phi_next * grad(mu_next))) * dx) - dt * lambda_next * assemble((div(u_now * rho_now) * omega_now + dot(u_next_1, rho_next * grad(omega_next))) * dx)
        zeta = newton_armijo(
            func = lambda zeta: float(Constant(A_zeta * zeta ** 2 + B_zeta * zeta + C_zeta)),
            d_func = lambda zeta: float(Constant(2 * A_zeta * zeta + B_zeta)),
            init_guess=1
        )
        zeta_next.assign(zeta)
        # compute u_next p_next:
        up_next.assign(up_next_1 + zeta_next * up_next_2)
        ##  solve step 2 (end)

        if "debug" in options and options["debug"]:
            print(f"u_now = {norm(u_now)}, p_now = {norm(p_now)}, zeta_now = {float(zeta_now)}, zeta_next = {float(zeta_next)}") # debug
            print(f"exact_u = {norm(pde.exact_u)}, u_next = {norm(u_next)}, exact_p = {norm(pde.exact_p)}, p_next = {norm(p_next)}") # debug
        # up_next.sub(0).interpolate(pde.exact_u) # debug
        # up_next.sub(1).interpolate(pde.exact_p) # debug

        if "energyflag" in options and options["energyflag"]:
            with open(f"data/{options["savename"]}.txt", 'a') as f:
                f.write(f"{float(pde.t)},{assemble(inner(phi_next,1) * dx)},{assemble(inner(rho_next, 1) * dx)},{float((pde.S_phi / pde.epsilon ** 2 + pde.gamma_1) * norm(phi_next - phi_now) ** 2)},{float((pde.S_rho / pde.eta ** 2 + pde.gamma_2) * norm(rho_next - rho_now) ** 2)},{float(lambda_now) ** 2},{float(zeta_now) ** 2},{cal_energy(pde, phi_next, rho_next, u_next)}\n")

        # prepare for next step
        up_now.assign(up_next)
        phimu_now.assign(phimu_next)
        rhoomega_now.assign(rhoomega_next)
        lambda_now.assign(lambda_next)
        zeta_now.assign(zeta_next)

        if "vtksave" in options and options["vtksave"]:
            outfile.write(phi, mu, rho, omega, u, p)

        # 只有rank 0打印进度
        if rank == 0:
            PrintProgressBar(step + 1, num_steps, start_time)


    # compute errors
    errors = []
    if "return_errors" in options and options["return_errors"]:
        l2_error_phi = errornorm(pde.exact_phi, phimu_next.sub(0), "L2", mesh=pde.mesh)
        l2_error_mu = errornorm(pde.exact_mu, phimu_next.sub(1), "L2", mesh=pde.mesh)
        l2_error_rho = errornorm(pde.exact_rho, rhoomega_next.sub(0), "L2", mesh=pde.mesh)
        l2_error_omega = errornorm(pde.exact_omega, rhoomega_next.sub(1), "L2", mesh=pde.mesh)
        l2_error_u = errornorm(pde.exact_u, up_next.sub(0), "L2", mesh=pde.mesh)
        l2_error_p = errornorm(pde.exact_p, up_next.sub(1), "L2", mesh=pde.mesh)
        l2_error_lambda = float(abs(pde.exact_lambda - lambda_next))
        l2_error_zeta = float(abs(pde.exact_zeta - zeta_next))
        errors = [
            l2_error_phi,
            l2_error_mu,
            l2_error_rho,
            l2_error_omega,
            l2_error_u,
            l2_error_p,
            l2_error_lambda,
            l2_error_zeta
        ]

    results = [
        phimu_next.sub(0),
        phimu_next.sub(1),
        rhoomega_next.sub(0),
        rhoomega_next.sub(1),
        up_next.sub(0),
        up_next.sub(1),
        lambda_next,
        zeta_next
    ]

    return results, errors

def Solver_CHNS_Surfactant_2nd(pde, time_setting, up_func_space, phimu_func_space, rhoomega_func_space, ns_nullspace, options):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # unpacked the time settings
    start_time = time.time()
    start_T, end_T, dt = time_setting
    num_steps = ceil((end_T - start_T) / dt)
    pde.t.assign(start_T)

    # define phi_next_1 phi_next_2 phi_next phi_now mu_next_1 mu_next_2 mu_next mu_now
    phimu_next_1, phimu_next_2, phimu_prev, phimu_now, phimu_next = Function(phimu_func_space), Function(phimu_func_space), Function(phimu_func_space), Function(phimu_func_space), Function(phimu_func_space)
    phi_prev, mu_prev = split(phimu_prev)
    phi_now, mu_now = split(phimu_now)
    phi_next, mu_next = split(phimu_next)
    phi_next_1, mu_next_1 = split(phimu_next_1)
    phi_next_2, mu_next_2 = split(phimu_next_2)

    # define rho_next_1 rho_next_2 rho_next rho_now omega_next_1 omega_next_2 omega_next omega_now
    rhoomega_next_1, rhoomega_next_2, rhoomega_prev, rhoomega_now, rhoomega_next = Function(rhoomega_func_space), Function(rhoomega_func_space), Function(rhoomega_func_space), Function(rhoomega_func_space), Function(rhoomega_func_space)
    rho_prev, omega_prev = split(rhoomega_prev)
    rho_now, omega_now = split(rhoomega_now)
    rho_next, omega_next = split(rhoomega_next)
    rho_next_1, omega_next_1 = split(rhoomega_next_1)
    rho_next_2, omega_next_2 = split(rhoomega_next_2)

    # define up_next_1 up_next_2 up_next up_now
    up_next_1, up_next_2, up_prev, up_now, up_next = Function(up_func_space), Function(up_func_space), Function(up_func_space), Function(up_func_space), Function(up_func_space)
    u_prev, p_prev = split(up_prev)
    u_now, p_now = split(up_now)
    u_next, p_next = split(up_next)
    u_next_1, p_next_1 = split(up_next_1)
    u_next_2, p_next_2 = split(up_next_2)

    # define phi_star rho_star u_star
    phi_star = 2 * phi_now - phi_prev
    rho_star = 2 * rho_now - rho_prev
    u_star = 2 * u_now - u_prev

    # define lambda_next lambda_now lambda_prev zeta_next zeta_now zeta_prev
    lambda_next, lambda_now, lambda_prev = Constant(1.0), Constant(1.0), Constant(1.0)
    zeta_next, zeta_now, zeta_prev = Constant(1.0), Constant(1.0), Constant(1.0)

    # do one step 1st solver
    phimu_prev.sub(0).interpolate(pde.init_phi)
    phimu_prev.sub(1).interpolate(pde.init_mu)
    rhoomega_prev.sub(0).interpolate(pde.init_rho)
    rhoomega_prev.sub(1).interpolate(pde.init_omega)
    up_prev.sub(0).interpolate(pde.init_u)
    up_prev.sub(1).interpolate(pde.init_p)

    temp_time_setting = (start_T, dt, dt)
    results, _ = Solver_CHNS_Surfactant_1st(pde, temp_time_setting, up_func_space, phimu_func_space, rhoomega_func_space, ns_nullspace, options)
    phi, mu, rho, omega, u, p, _lambda, zeta = results
    phimu_now.sub(0).assign(phi)
    phimu_now.sub(1).assign(mu)
    rhoomega_now.sub(0).assign(rho)
    rhoomega_now.sub(1).assign(omega)
    up_now.sub(0).assign(u)
    up_now.sub(1).assign(p)
    lambda_now.assign(_lambda)
    zeta_now.assign(zeta)

    bc_u_up = DirichletBC(up_func_space.sub(0), pde.bc_u_up, 4) if hasattr(pde,"bc_u_up") else None
    bc_u_down = DirichletBC(up_func_space.sub(0), pde.bc_u_down, 3) if hasattr(pde, "bc_u_down") else None
    bc_u_left = DirichletBC(up_func_space.sub(0), pde.bc_u_left, 1) if hasattr(pde,"bc_u_left") else None
    bc_u_right = DirichletBC(up_func_space.sub(0), pde.bc_u_right, 2) if hasattr(pde,"bc_u_right") else None
    bc_u_all = DirichletBC(up_func_space.sub(0), pde.bc_u_all, "on_boundary") if hasattr(pde,"bc_u_all") else None
    bc_u0 = DirichletBC(up_func_space.sub(0), Constant([0] * up_func_space.mesh().geometric_dimension()), "on_boundary")

    # set trival and test functions
    phi_trial, mu_trial = TrialFunctions(phimu_func_space)
    phi_test, mu_test = TestFunctions(phimu_func_space)

    rho_trial, omega_trial = TrialFunctions(rhoomega_func_space)
    rho_test, omega_test = TestFunctions(rhoomega_func_space)

    u_trial, p_trial = TrialFunctions(up_func_space)
    u_test, p_test = TestFunctions(up_func_space)

    # step 1 (I): solve phi_next_1 phi_next_2 mu_next_1 mu_next_2
    equa_phi_L1 = (
            3 * inner(phi_trial, phi_test) * dx
            + 2 * dt * pde.M_phi * inner(grad(mu_trial), grad(phi_test)) * dx
    )
    equa_phi_L2 = (
            pde.gamma_1 * inner(grad(phi_trial), grad(mu_test)) * dx
            + pde.S_phi / pde.epsilon ** 2 * inner(phi_trial, mu_test) * dx
            - inner(mu_trial, mu_test) * dx
    )

    if pde.rhs_phi is not None:
        equa_phi_R1 = (
                inner(4 * phi_now - phi_prev, phi_test) * dx
                + 2 * dt * inner(pde.rhs_phi, phi_test) * dx
        )
    else:
        equa_phi_R1 = (
                inner(4 * phi_now - phi_prev, phi_test) * dx
        )
    equa_phi_R2 = (
            pde.S_phi / pde.epsilon ** 2 * inner(phi_star, mu_test) * dx
    )
    equa_phi_R3 = (
            - 2 * dt * inner(div(u_star * phi_star), phi_test) * dx
    )
    equa_phi_R4 = (
            - inner(func_h(pde, phi_star, rho_star), mu_test) * dx
    )

    # step 1 (II): solve rho_next_1 rho_next_2 omega_next_1 omega_next_2
    equa_rho_L1 = (
            3 * inner(rho_trial, rho_test) * dx
            + 2 * dt * pde.M_rho * inner(grad(omega_trial), grad(rho_test)) * dx
    )
    equa_rho_L2 = (
            pde.gamma_2 * inner(grad(rho_trial), grad(omega_test)) * dx
            + pde.S_rho / pde.eta ** 2 * inner(rho_trial, omega_test) * dx
            - inner(omega_trial, omega_test) * dx
    )

    if pde.rhs_rho is not None:
        equa_rho_R1 = (
                inner(4 * rho_now - rho_prev, rho_test) * dx
                + 2 * dt * inner(pde.rhs_rho, rho_test) * dx
        )
    else:
        equa_rho_R1 = (
                inner(4 * rho_now - rho_prev, rho_test) * dx
        )
    equa_rho_R2 = (
            pde.S_rho / pde.eta ** 2 * inner(rho_star, omega_test) * dx
    )
    equa_rho_R3 = (
            - 2 * dt * inner(div(u_star * rho_star), rho_test) * dx
    )
    equa_rho_R4 = (
            - inner(func_q(pde, phi_star, rho_star), omega_test) * dx
    )

    # step 2: solve up_next_1 up_next_2
    equa_ns_L = (
        3 * inner(u_trial,u_test) * dx
        + (2 * dt / pde.Re) * inner(grad(u_trial),grad(u_test)) * dx
        - 2 * dt * inner(p_trial, div(u_test)) * dx
        + inner(div(u_trial),p_test) * dx
    )

    if pde.rhs_ns is not None:
        equa_ns_R1 = (
            inner(4 * u_now - u_prev, u_test) * dx
            + 2 * dt * inner(pde.rhs_ns - lambda_next * (phi_next * grad(mu_next) + rho_next * grad(omega_next)), u_test) * dx
        )
    else:
        equa_ns_R1 = (
            inner(4 * u_now - u_prev, u_test) * dx
            - 2 * dt * inner(lambda_next * (phi_next * grad(mu_next) + rho_next * grad(omega_next)), u_test) * dx
        )
    equa_ns_R2 = (
        - 2 * dt * inner(dot(grad(u_star), u_star),u_test) * dx
    )

    phi_A = assemble(equa_phi_L1 + equa_phi_L2)
    rho_A = assemble(equa_rho_L1 + equa_rho_L2)
    nullspace_rho = None
    nullspace_phi = None
    phi_solver = LinearSolver(phi_A, solver_parameters=PHI_SOLVER_PARAMS, nullspace=nullspace_phi)
    rho_solver = LinearSolver(rho_A, solver_parameters=PHI_SOLVER_PARAMS, nullspace=nullspace_rho)

    # step solver
    if "vtksave" in options and options["vtksave"]:
        phi, mu = phimu_now.subfunctions
        rho, omega = rhoomega_now.subfunctions
        u, p = up_now.subfunctions
        phi.rename("phi")
        mu.rename("mu")
        rho.rename("rho")
        omega.rename("omega")
        u.rename("u")
        p.rename("p")
        outfile = VTKFile(f"data/{options["savename"]}.pvd")
    for step in range(1, num_steps):
        next_t = start_T + (step + 1) * dt
        pde.t.assign(next_t)

        ##  solve step 1 (start):
        phi_b1 = assemble(equa_phi_R1 + equa_phi_R2)
        phi_solver.solve(phimu_next_1, phi_b1)

        phi_b2 = assemble(equa_phi_R3 + equa_phi_R4)
        phi_solver.solve(phimu_next_2, phi_b2)

        rho_b1 = assemble(equa_rho_R1 + equa_rho_R2)
        rho_solver.solve(rhoomega_next_1, rho_b1)

        rho_b2 = assemble(equa_rho_R3 + equa_rho_R4)
        rho_solver.solve(rhoomega_next_2, rho_b2)

        A_lambda = 3 * pde.kappa_lambda - 3 * assemble((func_h(pde, phi_star, rho_star) * phi_next_2 + func_q(pde, phi_star, rho_star) * rho_next_2) * dx)
        B_lambda = - assemble((func_h(pde, phi_star, rho_star) * (3 * phi_next_1 - 4 * phi_now + phi_prev) + func_q(pde, phi_star, rho_star) * (3 * rho_next_1 - 4 * rho_now + rho_prev)) * dx)
        C_lambda = pde.kappa_lambda * (-4 * lambda_now ** 2 + lambda_prev ** 2) + assemble((-4 * func_F_non(pde, phi_now, rho_now) + func_F_non(pde, phi_prev, rho_prev)) * dx)
        _lambda = newton_armijo(
            func=lambda _lambda: float(Constant(3 * assemble(func_F_non(pde, phi_next_1 + _lambda * phi_next_2, rho_next_1 + _lambda * rho_next_2) * dx) + A_lambda * _lambda ** 2 + B_lambda * _lambda + C_lambda)),
            d_func=lambda _lambda: float(Constant(3 * func_dlambda_int_F_non(pde, phi_next_1, phi_next_2, rho_next_1, rho_next_2, _lambda) + 2 * A_lambda * _lambda + B_lambda)),
            init_guess=1
        )
        lambda_next.assign(_lambda)
        # compute phi_next mu_next:
        phimu_next.assign(phimu_next_1 + lambda_next * phimu_next_2)
        # compute rho_next omega_next:
        rhoomega_next.assign(rhoomega_next_1 + lambda_next * rhoomega_next_2)
        ##  solve step 1 (end)

        if "debug" in options and options["debug"]:
            print(f"phi_now = {norm(phi_now)}, mu_now = {norm(mu_now)}, lambda_now = {float(lambda_now)}, lambda_next = {float(lambda_next)}") # debug
            print(f"exact_phi = {norm(pde.exact_phi)}, phi_next = {norm(phi_next)}, exact_mu = {norm(pde.exact_mu)}, mu_next = {norm(mu_next)}") # debug
            print(f"rho_now = {norm(rho_now)}, omega_now = {norm(omega_now)}, lambda_now = {float(lambda_now)}, lambda_next = {float(lambda_next)}") # debug
            print(f"exact_rho = {norm(pde.exact_rho)}, rho_next = {norm(rho_next)}, exact_omega = {norm(pde.exact_omega)}, omega_next = {norm(omega_next)}") # debug
        # phimu_next.sub(0).interpolate(pde.exact_phi) # debug
        # phimu_next.sub(1).interpolate(pde.exact_mu) # debug
        # rhoomega_next.sub(0).interpolate(pde.exact_rho) # debug
        # rhoomega_next.sub(1).interpolate(pde.exact_omega) # debug

        ##  solve step 2 (start):
        bcs = [bc for bc in [bc_u_up, bc_u_down, bc_u_left, bc_u_right, bc_u_all] if bc is not None] or None
        solve(equa_ns_L == equa_ns_R1, up_next_1, bcs=bcs, nullspace=ns_nullspace, solver_parameters=NS_SOLVER_PARAMS)
        solve(equa_ns_L == equa_ns_R2, up_next_2, bcs=bc_u0, nullspace=ns_nullspace, solver_parameters=NS_SOLVER_PARAMS)

        A_zeta = 3 * pde.kappa_zeta + 1.5 * assemble(inner(u_next_2, u_next_2) * dx) + 2 * dt * (1 / pde.Re) * assemble(inner(grad(u_next_2),grad(u_next_2)) * dx)
        B_zeta = - assemble(dot(3 * u_next_1 - 4 * u_now + u_prev, u_next_2) * dx) - 2 * dt * assemble((lambda_next * dot(phi_next * grad(mu_next) + rho_next * grad(omega_next),u_next_2) + dot(u_next_1,dot(grad(u_star), u_star))) * dx)
        C_zeta = pde.kappa_zeta * (-4 * zeta_now ** 2 + zeta_prev ** 2) - 2 * assemble(inner(u_next_1 - u_now, u_next_1 - u_now) * dx) + 0.5 * assemble(inner(u_next_1 - u_prev, u_next_1 - u_prev) * dx) - 2 * dt * lambda_next * assemble((div(u_star * phi_star) * mu_next + dot(u_next_1, phi_next * grad(mu_next))) * dx) - 2 * dt * lambda_next * assemble((div(u_star * rho_star) * omega_now + dot(u_next_1, rho_next * grad(omega_next))) * dx)
        zeta = newton_armijo(
            func = lambda zeta: float(Constant(A_zeta * zeta ** 2 + B_zeta * zeta + C_zeta)),
            d_func = lambda zeta: float(Constant(2 * A_zeta * zeta + B_zeta)),
            init_guess=1
        )
        zeta_next.assign(zeta)
        # compute u_next p_next:
        up_next.assign(up_next_1 + zeta_next * up_next_2)
        ##  solve step 2 (end)

        if "debug" in options and options["debug"]:
            print(f"u_now = {norm(u_now)}, p_now = {norm(p_now)}, zeta_now = {float(zeta_now)}, zeta_next = {float(zeta_next)}") # debug
            print(f"exact_u = {norm(pde.exact_u)}, u_next = {norm(u_next)}, exact_p = {norm(pde.exact_p)}, p_next = {norm(p_next)}") # debug
        # up_next.sub(0).interpolate(pde.exact_u) # debug
        # up_next.sub(1).interpolate(pde.exact_p) # debug

        # prepare for next step
        up_prev.assign(up_now)
        rhoomega_prev.assign(rhoomega_now)
        phimu_prev.assign(phimu_now)
        lambda_prev.assign(lambda_now)
        zeta_prev.assign(zeta_now)

        up_now.assign(up_next)
        rhoomega_now.assign(rhoomega_next)
        phimu_now.assign(phimu_next)
        lambda_now.assign(lambda_next)
        zeta_now.assign(zeta_next)

        if "vtksave" in options and options["vtksave"]:
            outfile.write(phi, mu, rho, omega, u, p)
        if "energyflag" in options and options["energyflag"]:
            with open(f"data/{options["savename"]}.txt", 'a') as f:
                f.write(f"{float(pde.t)},{assemble(inner(phi_now, 1) * dx)},{assemble(inner(rho_now, 1) * dx)},{float((pde.S_phi / pde.epsilon ** 2 + pde.gamma_1) * norm(phi_now - phi_prev) ** 2)},{float((pde.S_rho / pde.eta ** 2 + pde.gamma_2) * norm(rho_now - rho_prev) ** 2)},{float(lambda_now) ** 2},{float(zeta_now) ** 2},{cal_energy(pde, phi_now, rho_now, u_now)}\n")

        # 只有rank 0打印进度
        if rank == 0:
            PrintProgressBar(step, num_steps - 1, start_time)

    # compute errors
    errors = []
    if "return_errors" in options and options["return_errors"]:
        l2_error_phi = errornorm(pde.exact_phi, phimu_next.sub(0), "L2", mesh=pde.mesh)
        l2_error_mu = errornorm(pde.exact_mu, phimu_next.sub(1), "L2", mesh=pde.mesh)
        l2_error_rho = errornorm(pde.exact_rho, rhoomega_next.sub(0), "L2", mesh=pde.mesh)
        l2_error_omega = errornorm(pde.exact_omega, rhoomega_next.sub(1), "L2", mesh=pde.mesh)
        l2_error_u = errornorm(pde.exact_u, up_next.sub(0), "L2", mesh=pde.mesh)
        l2_error_p = errornorm(pde.exact_p, up_next.sub(1), "L2", mesh=pde.mesh)
        l2_error_lambda = float(abs(pde.exact_lambda - lambda_next))
        l2_error_zeta = float(abs(pde.exact_zeta - zeta_next))
        errors = [
            l2_error_phi,
            l2_error_mu,
            l2_error_rho,
            l2_error_omega,
            l2_error_u,
            l2_error_p,
            l2_error_lambda,
            l2_error_zeta
        ]

    results = [
        phimu_next.sub(0),
        phimu_next.sub(1),
        rhoomega_next.sub(0),
        rhoomega_next.sub(1),
        up_next.sub(0),
        up_next.sub(1),
        lambda_next,
        zeta_next
    ]

    return results, errors