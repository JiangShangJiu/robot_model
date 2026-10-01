"""离线周期平均：按时间相位插值，绝不跨随机种子或不同对象平均。"""
from dataclasses import dataclass
import numpy as np
from .dataset import MotionData
from ..utils.validation import integer


@dataclass(frozen=True)
class PeriodAverage:
    data: MotionData
    cycles: int
    standard_deviation: dict

    def summary(self):
        return {
            'cycles': self.cycles, 'phase_points': len(self.data),
            'phase_step_s': float(self.data.t[1]-self.data.t[0]),
            'cycle_scatter_rms': {
                key: np.sqrt(np.mean(value**2, axis=0)).tolist()
                for key, value in self.standard_deviation.items()
            },
            'note': '周期间离散度包含噪声、非周期扰动和漂移；不是独立样本标准误差。',
        }


def average_periods(data, period, *, cycles=None):
    """对连续、均匀采样的原始测量按相同相位平均，然后再滤波/去低速。

    周期不必是采样步长的整数倍。只在所有周期共同覆盖的相位网格插值，
    不外推、不把首尾强行接起来；每周期共同覆盖至少 95% 的相位。
    """
    t = np.asarray(data.t, float)
    if not np.isfinite(period) or period <= 0 or t.ndim != 1 or len(t) < 4:
        raise ValueError('周期与采样时间无效')
    dt = np.diff(t)
    if not np.isfinite(t).all() or np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-10):
        raise ValueError('周期平均必须在去低速之前使用连续、均匀采样数据')
    step = float(dt[0])
    if period < 20*step-1e-10:
        raise ValueError('每周期至少需要 20 个采样点，以覆盖至少 95% 的相位')
    if cycles is None:
        cycles = int(np.floor((t[-1]-t[0])/period))+1
        if t[-1]-t[0]-(cycles-1)*period < .95*period:
            cycles -= 1
    integer(cycles, 'cycles', 2)
    last_phase = min(period-step, t[-1]-t[0]-(cycles-1)*period)
    if last_phase < .95*period-1e-9:
        raise ValueError('数据不足以覆盖指定数量的完整周期')
    points = int(np.floor(period/step+1e-7))
    phases = np.arange(points)*(period/points)
    phases = phases[phases <= last_phase+1e-10]
    targets = t[0]+np.arange(cycles)[:, None]*period+phases[None, :]
    means, deviations = {}, {}
    for key in ('q', 'dq', 'ddq', 'tau'):
        values = np.asarray(getattr(data, key), float)
        if values.shape != (len(t), data.dof) or not np.isfinite(values).all():
            raise ValueError(f'{key} 的形状或数值无效')
        blocks = np.stack([np.interp(targets.ravel(), t, values[:, j]).reshape(targets.shape)
                           for j in range(data.dof)], axis=-1)
        means[key] = blocks.mean(axis=0)
        deviations[key] = blocks.std(axis=0, ddof=1)
    return PeriodAverage(MotionData(t[0]+phases, **means), cycles, deviations)
