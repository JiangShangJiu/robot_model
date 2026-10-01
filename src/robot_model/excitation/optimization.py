"""固定基频/谐波数，对傅里叶系数做受约束的局部 SLSQP 优化。"""
from dataclasses import dataclass
import numpy as np
from ..data.dataset import MotionData
from .screening import regressor_quality, trajectory_record
from ..utils.validation import integer, vector
from .trajectory import FourierTrajectory


@dataclass(frozen=True)
class ExcitationOptimization:
    trajectory: FourierTrajectory
    report: dict


def optimize_excitation(regressor_func, initial, *, limits, velocity_limits,
                        acceleration_limits, maxiter=20, samples=96, margin=.1):
    """优化 log(cond(W))，用谐波幅值和约束整个连续周期。

    保留可行初值和优化过程中更好的可行解；达到迭代上限不冒充已收敛。
    最后在更密、错开相位的网格验证，若未改善则回退到初值。
    仅约束运动学，不替代闭环力矩/跟踪检查，不承诺全局最优。
    """
    from scipy.optimize import minimize
    integer(maxiter, 'maxiter', 1)
    integer(samples, 'samples', 8)
    n, h = initial.a.shape
    if samples <= 2*h or not 0 <= margin < 1:
        raise ValueError('采样点必须超过两倍谐波数，margin 必须在 [0,1)')
    lo, hi = vector(limits[0], n, 'lower'), vector(limits[1], n, 'upper')
    velocity = vector(velocity_limits, n, 'velocity', positive=True)
    acceleration = vector(acceleration_limits, n, 'acceleration', positive=True)
    start = initial.constrained(lower=lo, upper=hi, velocity=velocity,
                                acceleration=acceleration, margin=margin)
    room = np.minimum(start.q0-lo, hi-start.q0)*(1-margin)
    vroom, aroom = velocity*(1-margin), acceleration*(1-margin)
    wk = 2*np.pi*start.base_freq*np.arange(1, h+1)
    size = n*h
    def unpack(x):
        return FourierTrajectory(start.q0.copy(), x[:size].reshape(n, h),
                                 x[size:].reshape(n, h), start.base_freq)
    def constraints(x):
        a, b = x[:size].reshape(n, h), x[size:].reshape(n, h)
        amp = np.hypot(a, b)
        return np.concatenate([1-np.sum(amp/wk, axis=1)/room,
                               1-np.sum(amp, axis=1)/vroom,
                               1-np.sum(amp*wk, axis=1)/aroom])
    def quality(traj, count, shifted=False):
        t = (np.arange(count)+(0.37 if shifted else 0))*traj.period/count
        q, dq, ddq = traj.evaluate(t)
        return regressor_quality(regressor_func, MotionData(t, q, dq, ddq, np.zeros_like(q)))
    x0 = np.concatenate([start.a.ravel(), start.b.ravel()])
    before = quality(start, samples)
    if not np.isfinite(before['condition_number']):
        raise ValueError('优化初值的解析回归矩阵秩不足')
    best_x, best_value = x0.copy(), np.log(before['condition_number'])
    evaluations = 0
    def objective(x):
        nonlocal best_x, best_value, evaluations
        evaluations += 1
        cond = quality(unpack(x), samples)['condition_number']
        value = np.log(cond) if np.isfinite(cond) else 1e6
        if np.min(constraints(x)) >= 0 and value < best_value:
            best_x, best_value = x.copy(), value
        return value
    def retain_feasible_iterate(x):
        # SLSQP 的中间迭代可能略微越界；重新缩放后独立评分，绝不把
        # 不可行点的较低条件数当作可行解的成绩。
        feasible = unpack(x).constrained(lower=lo, upper=hi, velocity=velocity,
                                          acceleration=acceleration, margin=margin)
        safe_x = np.concatenate([feasible.a.ravel(), feasible.b.ravel()])
        if np.min(constraints(safe_x)) < 0:
            safe_x *= 1-1e-12
        objective(safe_x)
    bound = np.tile(np.repeat(vroom, h), 2)
    result = minimize(objective, x0, method='SLSQP', bounds=list(zip(-bound, bound)),
                      constraints={'type': 'ineq', 'fun': constraints},
                      callback=retain_feasible_iterate,
                      options={'maxiter': maxiter, 'ftol': 1e-6})
    if np.isfinite(result.x).all():
        retain_feasible_iterate(result.x)
    candidate = unpack(best_x)
    dense_before = quality(start, samples*3, True)
    dense_after = quality(candidate, samples*3, True)
    adopted = dense_after['condition_number'] < dense_before['condition_number']*(1-1e-8)
    selected = candidate if adopted else start
    return ExcitationOptimization(selected, {
        'method': 'SLSQP', 'converged': bool(result.success), 'message': str(result.message),
        'iterations': int(result.nit), 'evaluations': evaluations, 'maxiter': maxiter,
        'objective': 'log condition number, fixed base frequency and harmonic count',
        'samples': samples, 'validation_samples': samples*3,
        'before': dense_before, 'after': dense_after if adopted else dense_before,
        'adopted': bool(adopted), 'constraint_min_slack': float(np.min(constraints(best_x if adopted else x0))),
        'initial': trajectory_record(start), 'trajectory': trajectory_record(selected),
    })
