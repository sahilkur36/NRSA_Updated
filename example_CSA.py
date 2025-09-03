import time
from math import pi, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis import ConstantStrengthAnalysis


def material_definition(
    Ti: float,
    mass: float,
    Sa_5pct: float,
    Sa_spc: float,
    damping: float,
    scaling_factor: float,
    *args: float
) -> tuple[dict[str, tuple | float], float, float, float]:
    """该函数传入SDOF体系的周期、质量等信息，用户需通过`args`获取预先定义的相关控制参  
    数，并计算OpenSees材料的具体参数，并返回规定的变量。 
    
    Args:
        Ti (float): 周期点
        mass (float): 质量
        Sa_5pct (float): 无缩放，5%阻尼比的地震动弹性谱加速度
        Sa_spc (float): 无缩放，分析用阻尼比的地震动弹性谱加速度
        damping (float): 阻尼比
        scaling_factor (float): 地震动缩放系数系数
        Args (float): 定义opensees材料所需的相关参数，一般建议取为无量纲系数，并以此
          计算定义材料所需的直接参数，如`Cy`和`alpha`

    Returns:
        tuple[dict[str, tuple | float], float, float, float]: OpenSees材料定义格式  

    Note-1:
    -------
    OpenSees材料定义格式为{`材料名`: (参数1, 参数2, ...)}，不包括材料编号。  
    例如：  
    >>> ops_paras = {'Steel01': (Fy, E, b)}

    其中，`Fy`，`E`和`b`应直接幅值或通过`Ti`、`mass`和`Sa_spc`计算得到。  
    当需要使用多个材料进行并联时，可在`ops_paras`中定义多个材料。  
    例如：  
    >>> ops_paras = {'Steel01': (Fy, E1, b), 'Elastic': E2}
    注：当材料参数只有一个时，可省略括号
    
    Note-2:
    -------
    返回值中，`ops_paras`为OpenSees的材料参数，`yield_disp`为屈服位移，用于计算延性  
    需求，`initial_stiffness`为初始刚度，用于计算P-Delta效应引起的负刚度(当`thetaD=0`  
    时不考虑P-Delta效应)，`damping_coef`为阻尼系数(不是阻尼比)，可基于传入的`damping`  
    和`mass`进行计算。
    """
    # ===========================================================
    # --------------- ↓ 用户只能更改以下代码 ↓ --------------------
    Cy, alpha = args
    E = (2 * pi / Ti) ** 2 * mass
    Fy = mass * 9800 * Cy
    ops_paras = {'Steel01': (Fy, E, alpha), 'Elastic': 1}
    yield_disp, initial_stiffness = Fy / E, E
    omega = 2 * pi / Ti
    damping_coef = 2 * mass * damping * omega
    # --------------- ↑ 用户只能更改以上代码 ↑ --------------------
    # ===========================================================
    return ops_paras, yield_disp, initial_stiffness, damping_coef


if __name__ == "__main__":
    Cy_ls = [0.5, 0.4]  # 可以为一个数或一个列表
    alpha_ls = [0, 0.02]
    time_start = time.time()
    T = np.arange(0.02, 6, 0.02)
    material_paras: dict[str, float] = {
        'Cy': Cy_ls,  # A single value or a list of values can be used
        'alpha': alpha_ls
    }  # 材料定义所需参数，键名可自定义，字典长度应与material_definition函数中args参数个数一致
    model = ConstantStrengthAnalysis(f'Test')
    model.set_working_directory(f'./results_CSA/', folder_exists='delete')
    model.analysis_settings(T, material_definition, material_paras, damping=0.05, thetaD=0)
    model.select_ground_motions('./data/GMs', ['Northridge', 'Kobe'], suffix='.txt')
    code_spec = np.loadtxt('./data/DBE_spec.txt')
    model.scale_ground_motions('b', 1, code_spec, plot=True)
    model.running_settings(parallel=2, auto_quit=True, hidden_prints=True, show_monitor=True)
    model.run()
    time_end = time.time()
    print(f'Elapsed time: {time_end - time_start:.2f}')
    
    # Ductility demand spectrum
    g = 9800
    plt.figure(figsize=(12, 8))
    for i, gm in enumerate(['Northridge', 'Kobe']):
        plt.subplot(2, 1, i + 1)
        for j, Cy in enumerate(Cy_ls):
            for j, alpha in enumerate(alpha_ls):
                res = pd.read_csv(f'./results_CSA/results_{Cy}_{alpha}/{gm}.csv')
                T = res['T']
                miu = res['miu']
                plt.plot(T, miu, label=f'Cy={Cy}, alpha={alpha}')
        plt.xlabel('Period (s)')
        plt.ylabel('Ductility')
        plt.xlim(0, 6)
        plt.ylim(0)
        plt.title(f'Ductility Damend Spectrum ({gm})')
        plt.legend()
    plt.tight_layout()
    plt.show()
