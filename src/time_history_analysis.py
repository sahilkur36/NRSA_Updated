import os, json
import time
import traceback
import multiprocessing
import importlib
from pathlib import Path
from typing import Callable
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import SOLVER_TYPING
from .utils import SDOFHelper, SDOFError


def time_history_analysis(*args, **kwargs):
    queue: multiprocessing.Queue = args[-4]
    hidden_prints: bool = args[-5]
    try:
        with SDOFHelper(getTime=False, suppress=hidden_prints):
            _time_history_analysis(*args, **kwargs)
    except Exception as error:
        tb = traceback.format_exc()
        queue.put({'d': (error, tb)})
        return

def _time_history_analysis(
    wkdir: Path,
    subfolder: str,
    Ti: float,
    material_function: Callable[[float, float, float, float],
                                tuple[dict[str, tuple | float], float, float]],
    material_paras: tuple[float],
    damping: float,
    thetaD: float,
    mass: float,
    height: float,
    GM_name: str,
    gm_shm_name: str,
    NPTS: int,
    scaling_factor: float,
    dt: float,
    fv_duration: float,
    fv_factor: float,
    periods_shm_name: str,
    N_PERIOD: int,
    Sa_shm_name: str,
    solver: SOLVER_TYPING,
    hidden_prints: bool,
    queue: multiprocessing.Queue,
    stop_event,
    pause_event,
    lock,
    **kwargs
):
    """等延性分析迭代函数

    Args:
        wkdir (Path): 工作路径
        subfolder (str): 子文件夹名称
        Ti (np.ndarray): 周期点
        material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): opensees格式材料定义函数
        material_paras (tuple[float]): 滞回模型控制参数，用于输入material_function中获得材料参数
        damping (float): 阻尼比
        thetaD (float): P-Delta系数
        mass (float): 质量
        height (float): 高度
        GM_name (str): 地震动名称
        gm_shm_name (str): 地震动共享内存名
        NPTS (int): 时间步数
        scaling_factor (float): 地震动时缩放系数
        dt (float): 时间步长
        fv_duration (float): 自由振动持续时间
        periods_shm_name (str): 周期序列共享内存名
        N_PERIOD (int): 周期数
        Sa_shm_name (str): 加速度反应谱共享内存名
        solver (SOLVER_TYPES): SODF求解器类型
        hidden_prints (bool): 是否屏蔽输出
        queue (multiprocessing.Queue): 进程通信
        stop_event: 停止事件
        pause_event: 暂停事件
        lock: 锁，文件读写时使用
        kwargs: 求解器参数
    """
    periods_shm = SharedMemory(name=periods_shm_name)
    periods = np.ndarray(shape=(N_PERIOD,), dtype=np.dtype('float64'), buffer=periods_shm.buf).copy()
    periods_shm.close()
    Sa_shm = SharedMemory(name=Sa_shm_name)
    Sa_ls = np.ndarray(shape=(N_PERIOD,), dtype=np.dtype('float64'), buffer=Sa_shm.buf).copy()
    Sa_shm.close()
    get_Sa = interp1d(periods, Sa_ls, kind='linear', fill_value='extrapolate')
    gm_shm = SharedMemory(name=gm_shm_name)
    th = np.ndarray(shape=(NPTS,), dtype=np.dtype('float64'), buffer=gm_shm.buf).copy()
    gm_shm.close()
    num_ana = 1
    results = pd.DataFrame(None, columns=['T', 'E', 'Fy', 'uy', 'Sa', 'R', 'miu', 'maxDisp', 'maxVel', 'maxAccel', 'Ec', 'Ev', 'maxReaction', 'CD', 'CPD','resDisp', 'solving_converge'])
    package_name = __package__
    module_attr = {
        'newmark': 'newmark_solver',
        'ops_solver': 'ops_solver'
    }
    start_time = time.time()
    solving_converge = 1  # 求解是否收敛，如果有不收敛则变为False
    if stop_event.is_set():
        queue.put({'e': '中断计算'})
        return
    pause_event.wait()
    if Ti is not None:
        Sa = get_Sa(Ti)  # 弹性谱加速度
    else:
        Sa = None
    mat_paras, Fy, E = material_function(Ti, mass, Sa, *material_paras)
    uy = Fy / E
    if thetaD == 0:
        P = 0
    else:
        P = thetaD * E * height
    fv_duration = max(fv_duration, fv_factor * Ti)
    solver_paras = (Ti, th, dt, mat_paras, uy, fv_duration, scaling_factor, P, height, damping, mass)
    try:
        res: dict[str, float]
        res_th: tuple[np.ndarray, ...]
        if solver == 'auto':
            for module_name in ['newmark', 'ops_solver']:
                solver_func = importlib.import_module(f".{module_name}", package=package_name).__getattribute__(module_attr[module_name])
                res, res_th = solver_func(*solver_paras, record_res=True, **kwargs)
                if res['converge']:
                    break 
        elif solver == 'OPS':
            solver_func = importlib.import_module(f".ops_solver", package=package_name).__getattribute__('ops_solver')
            res, res_th = solver_func(*solver_paras, record_res=True, **kwargs)
        elif solver == 'Newmark-Newton':
            solver_func = importlib.import_module(f".newmark", package=package_name).__getattribute__('newmark_solver')
            res, res_th = solver_func(*solver_paras, record_res=True, **kwargs)
        else:
            raise SDOFError(f'Wrong solver name: {solver}')
    except:
        raise SDOFError('Solver error!')
    is_converged: bool = res['converge']  # 计算是否收敛
    if not is_converged:
        solving_converge = 0
        queue.put({'b': (GM_name, Ti, solver_paras[3:])})
        print(f'Warning: Unconverged analysis! ')
        print(f'Ground motion: {GM_name}, Period: {Ti}, dt: {dt}, solver_paras:', *solver_paras[3:])
        unconverged_res = {
            'period': Ti,
            'ground motion': GM_name,
            'dt': dt,
            'mat_paras': mat_paras,
            'uy': uy,
            'fv_duration': fv_duration,
            'scaling_factor': scaling_factor,
            'P': P,
            'height': height,
            'damping': damping,
            'mass': mass,
            'E': E,
            'Fy': Fy
        }
        lock.acquire()
        if not Path(wkdir / subfolder / f'warnings').exists():
            os.makedirs(wkdir / subfolder / f'warnings')
        json.dump(unconverged_res, open(wkdir / subfolder / f'warnings/c.json', 'w', encoding='utf8'), ensure_ascii=False, indent=4)
        lock.release()
    row = len(results)
    maxDisp = res['maxDisp']
    results.loc[row] = res
    results.loc[row, 'T'] = Ti
    results.loc[row, 'E'] = E
    results.loc[row, 'Fy'] = Fy
    results.loc[row, 'uy'] = Fy / E
    results.loc[row, 'Sa'] = Sa
    results.loc[row, 'R']   = mass * Sa * 9800 / Fy
    results.loc[row, 'miu'] = maxDisp / uy
    results.loc[row, 'solving_converge'] = solving_converge
    end_time = time.time()
    time_, ag_scaled, disp_th, vel_th, accel_th, Ec_th, Ev_th, CD_th, CPD_th, reaction_th, eleForce_th, dampingForce_th = res_th
    res_data = np.column_stack((time_, ag_scaled, disp_th, vel_th, accel_th, Ec_th, Ev_th, CD_th, CPD_th, reaction_th, eleForce_th, dampingForce_th))
    lock.acquire()
    results.to_csv(wkdir / subfolder / f'{GM_name}.csv', index=False)
    np.save(wkdir / subfolder / f'{GM_name}.npy', res_data)
    lock.release()
    queue.put({'a': (GM_name, num_ana, 1, None, solving_converge, start_time, end_time, None)})
    return None
