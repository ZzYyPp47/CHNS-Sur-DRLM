# -*- coding:utf-8 -*-
'''
@Author: ZYP
@Contact: 3137168510@qq.com
@Time: 2025/10/26 22:03
@File: UtilityFunctions.py
'''

import numpy as np
import time
import sys
import os


def CallTimeConvergence(solver, dt_list, karg):
    data = []
    start_T, end_T, _ = karg["time_setting"]
    name = karg["options"]["savename"]
    for dt in dt_list:
        karg["time_setting"] = (start_T, end_T, dt)
        karg["options"]["savename"] = name + f"_{dt}"
        _, errors = solver(**karg)
        data.append([dt, *errors])
    data = np.array(data)

    # 计算所有误差的收敛率（跳过dt列）
    n_errors = data.shape[1] - 1  # 总误差列数
    d_logs = np.log(data[:-1] / data[1:])
    ddt = d_logs[:, 0]
    rates = d_logs[:, 1:].T / ddt  # 所有误差的收敛率
    rates = np.insert(rates, 0, 0, axis=1)

    rate_data = np.column_stack([data[:, 0]] + [col for i in range(n_errors) for col in (data[:, i + 1], rates[i])])
    return rate_data

def PrintConvergenceTable(rate_data,headers):
    # 确定列数
    num_cols = len(rate_data[0])

    # 确定每一列的最大宽度
    col_widths = [0] * num_cols

    # 计算表头宽度
    for i in range(num_cols):
        col_widths[i] = max(col_widths[i], len(headers[i]))

    for row in rate_data:
        for i, value in enumerate(row):
            if i == 0:  # dt列
                formatted = f"{value:.4f}"
            elif i % 2 == 1:  # 误差列（奇数索引）
                formatted = f"{value:.4e}"
            else:  # 收敛率列（偶数索引，除了0）
                formatted = f"{value:.4f}"
            col_widths[i] = max(col_widths[i], len(formatted))

    print(f"\nTime Convergence Test:")
    # 打印表头
    header_line = "|"
    for i in range(num_cols):
        header_line += f" {headers[i]:^{col_widths[i]}} |"
    print(header_line)
    # 打印数据行
    for row_idx, row in enumerate(rate_data):
        data_line = "|"
        for i, value in enumerate(row):
            if i == 0:  # dt列
                formatted = f"{value:.4f}"
            elif i % 2 == 1:  # 误差列
                formatted = f"{value:.4e}"
            else:  # 收敛率列
                formatted = f"{value:.4f}"
            data_line += f" {formatted:^{col_widths[i]}} |"
        print(data_line)

def _format_time(seconds):
    """将秒数格式化为 HH:MM:SS 或 MM:SS"""
    if seconds < 0:
        return "0:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"

def _format_speed(speed):
    """
    将每步平均耗时格式化为速度字符串。
    若 speed < 1 秒，显示为 xx.xx it/s（每秒迭代数）；
    否则显示为 xx.xx s/it（每迭代秒数）。
    """
    if speed <= 0:
        return "0 it/s"
    if speed < 1:
        return f"{1 / speed:.2f} it/s"
    else:
        return f"{speed:.2f} s/it"

def PrintProgressBar(current, total, start_time, bar_length=30):
    """
    打印简洁的进度条（包含进度、时间、速度）。

    参数:
        current: 已完成步数 (int)
        total: 总步数 (int)
        start_time: 开始时间 (time.time())
        bar_length: 进度条长度（字符数）
    """
    if total <= 0:
        return

    progress = current / total
    percent = progress * 100
    elapsed = time.time() - start_time

    if current > 0:
        avg_time = elapsed / current  # 每步平均耗时
        remain = avg_time * (total - current)
    else:
        avg_time = 0
        remain = 0

    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)

    # 颜色（可选，不喜欢可以去掉 \033 代码）
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'

    # 构建输出字符串
    output = (f"\r{BLUE}[{bar}]{ENDC} "
              f"{GREEN}{current}/{total} - {percent:5.1f}% {ENDC} "
              f"{YELLOW}[{_format_time(elapsed)} < ETA: {_format_time(remain)}]{ENDC} "
              f"({_format_speed(avg_time)})")

    print(output, end="", flush=True, file=sys.stderr)

    if current == total:
        print()  # 完成时换行