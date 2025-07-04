import time
from math import pi, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis import TimeHistoryAnalysis


def material_definition(
    Ti: float,
    m: float,
    Sa: float,
    *args: float
) -> tuple[dict[str, tuple | float], float, float]:
    """给定周期点、质量、强度折减系数、弹性谱加速度，用户自定义opensees材料参数的计算方法  
    用户可修改传入的`args`参数和函数内部的计算过程，但不能修改函数的前两个传入参数(`Ti`、`m`和`Sa`)  
    返回值中的屈服力和弹性刚度将用于计算延性需求

    Args:
        Ti (float): 周期点
        m (float): 质量
        Sa (float): 弹性谱加速度(g)，等强度分析中，Sa为采用5%阻尼比的缩放后的谱加速度
        Args (float): 定义opensees材料所需的相关参数，一般建议取为无量纲系数，并以此计算定义材料所需的直接参数

    Returns:
        tuple[dict[str, tuple | float], float, float]: OpenSees材料定义格式

    Note:
    -----
    OpenSees材料定义格式为{`材料名`: (参数1, 参数2, ...)}，不包括材料编号。  
    例如：  
    >>> ops_paras = {'Steel01': (Fy, E, b)}

    其中，`Fy`，`E`和`b`应直接幅值或通过`Ti`、`m`和`Sa`计算得到。  
    当需要使用多个材料进行并联时，可在`ops_paras`中定义多个材料。  
    例如：  
    >>> ops_paras = {'Steel01': (Fy, E1, b), 'Elastic': E2}
    注：当材料参数只有一个时，可省略括号
    """
    # ===========================================================
    # --------------- ↓ 用户只能更改以下代码 ↓ --------------------
    Cy, alpha = args
    E = (2 * pi / Ti) ** 2 * m
    Fy = m * 9800 * Cy
    ops_paras = {'Steel01': (Fy, E, alpha)}
    yield_strength, initial_stiffness = Fy, E
    # --------------- ↑ 用户只能更改以上代码 ↑ --------------------
    # ===========================================================
    return ops_paras, yield_strength, initial_stiffness


if __name__ == "__main__":
    Ti = 0.5
    Cy_ls = [0.5, 1]  # 可以为一个数或一个列表
    alpha_ls = [0, 0.02]
    time_start = time.time()
    material_paras: dict[str, float] = {
        'Cy': Cy_ls,  # A single value or a list of values can be used
        'alpha': alpha_ls
    }  # 材料定义所需参数，键名可自定义，字典长度应与material_definition函数中args参数个数一致
    model = TimeHistoryAnalysis(f'Test_THA')
    model.set_working_directory(f'./results_THA', folder_exists='delete')
    model.analysis_settings(Ti, material_definition, material_paras, damping=0.05,
                            thetaD=0, fv_duration=30, fv_factor=30)
    model.select_ground_motions('./data/GMs', ['Northridge', 'Kobe'], suffix='.txt')
    code_spec = np.loadtxt('./data/DBE_spec.txt')
    model.scale_ground_motions('b', 1, code_spec, plot=True)
    model.running_settings(parallel=2, auto_quit=True, hidden_prints=True, show_monitor=True)
    model.run()
    time_end = time.time()
    print(f'Elapsed time: {time_end - time_start:.2f}')
    results = model.get_results(
        gm_name='Northridge',
        material_paras={'Cy': 0.5, 'alpha': 0.02},
        plot=True)
    time_, ag_scaled, disp_th, vel_th, accel_th, Ec_th, Ev_th, CD_th, CPD_th, reaction_th, eleForce_th, dampingForce_th = results.T
