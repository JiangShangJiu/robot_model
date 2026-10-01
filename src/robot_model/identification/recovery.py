# SPDX-License-Identifier: GPL-3.0-only
# 算法改编自 FrankaEmikaPandaDynModel/parameters_retrieval，
# 原作者 Claudio Gaz 与 Marco Cognetti，2019 年 8 月 2 日。
# 为 robot_model（2026）改写：Python/SciPy 实现、通用基参数映射、
# 支持名义初值、作动器项，结果带物理审计。
"""分两步恢复一组物理可行的连杆惯性参数。

保留参考算法的多起点退火、局部优化和递增*线性*物理罚项。
用 SciPy 优化器代替 MATLAB 优化器，用通用基参数映射代替写死的 Panda 表达式。
不依赖仿真器或实验模块，SciPy 只在使用时才导入。

返回的候选不等于恢复成功：要看 ``success``，以及求解器 / 物理 / 基参数
三组各自的诊断量。结果只是可行解之一，连杆参数一般不是唯一可辨识的。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..utils.validation import integer

_TENSOR_INDICES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_TENSOR_BASIS = np.zeros((6, 3, 3))
for _k, (_i, _j) in enumerate(_TENSOR_INDICES):
    _TENSOR_BASIS[_k, _i, _j] = _TENSOR_BASIS[_k, _j, _i] = 1.0


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _array(value, shape, name):
    try:
        a = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} 不能广播为 {shape}") from exc
    if not np.isfinite(a).all():
        raise ValueError(f"{name} 必须有限")
    return a


@dataclass(frozen=True)
class RecoveryConfig:
    """恢复用的参数界、优化器预算和独立验收条件。

    质量单位 kg；质心单位 m，在连杆坐标系下表达；惯量单位 kg*m^2，
    取质心处、在连杆坐标轴下表达。质量可给标量或 n 维向量；质心可给
    标量、xyz 向量或 n-by-3 数组。``penalty_weights`` 作用在线性 hinge
    罚项上：历史质量 / 惯量项沿用 kg / kg*m^2，新增球包络项无量纲。

    ``nominal_prior_weight`` 默认 0，仅对 m/c/I 加软先验，不惩罚作动器项。
    先验以名义质量、包络半径（未设置时为质心盒半对角线）、名义惯量
    Frobenius 范数归一化，详见结果里的 ``nominal_prior_scales``。
    权重相对于 ``base_error**2``，不要求名义参数等于真值。

    ``geometry_center`` / ``geometry_radius`` 必须同时给出；分别可广播
    为 (n, 3) / (n,)，坐标系和单位与质心一致。约束
    trace(I_com)/2 + m*||c-center||^2 <= m*radius^2 是选定球形质量分布
    包络的二阶矩约束，不能等同于逐点 CAD 网格碰撞或精确形状约束。
    几何验收公差作用于除以 m*radius^2 后的无量纲违反量。
    两个可选功能默认关闭。
    """
    mass_lower: Any = 0.001
    mass_upper: Any = 10.0
    com_lower: Any = (-0.5, -0.5, -0.5)
    com_upper: Any = (0.5, 0.5, 0.5)
    inertia_diagonal_lower: float = 0.0
    inertia_diagonal_upper: float = 1.0
    inertia_offdiagonal_bound: float = 1.0
    drive_inertia_upper: float = 1.0
    friction_upper: float = 5.0
    offset_bound: float = 5.0
    total_mass_lower: float | None = None
    total_mass_upper: float | None = None
    runs: int = 3
    penalty_weights: Any = (0.0, 10.0, 20.0, 30.0, 40.0)
    annealing_maxiter: int = 10
    local_maxiter: int = 500
    ftol: float = 1e-10
    seed: int = 0
    feasibility_tolerance: float = 1e-6
    max_base_error: float = 0.05
    nominal_prior_weight: float = 0.0
    geometry_center: Any = None
    geometry_radius: Any = None

    def __post_init__(self):
        for name in ('runs', 'annealing_maxiter', 'local_maxiter'):
            integer(getattr(self, name), name, 1)
        integer(self.seed, 'seed')
        for name in ('inertia_diagonal_upper', 'inertia_offdiagonal_bound',
                     'drive_inertia_upper', 'friction_upper', 'offset_bound',
                     'ftol', 'feasibility_tolerance', 'max_base_error'):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} 必须为有限正数')
        if (not np.isfinite(self.inertia_diagonal_lower)
                or not 0 <= self.inertia_diagonal_lower < self.inertia_diagonal_upper):
            raise ValueError('惯量对角线下界必须非负且小于上界')
        for name in ('total_mass_lower', 'total_mass_upper'):
            value = getattr(self, name)
            if value is not None and (not np.isscalar(value) or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} 必须为有限正数或 None')
        if (self.total_mass_lower is not None and self.total_mass_upper is not None
                and self.total_mass_lower > self.total_mass_upper):
            raise ValueError('总质量下界不得大于上界')
        if (not np.isscalar(self.nominal_prior_weight)
                or not np.isfinite(self.nominal_prior_weight) or self.nominal_prior_weight < 0):
            raise ValueError('nominal_prior_weight 必须为有限非负数')
        if (self.geometry_center is None) != (self.geometry_radius is None):
            raise ValueError('geometry_center 和 geometry_radius 必须同时提供')
        if self.geometry_radius is not None:
            radius = np.asarray(self.geometry_radius, dtype=float)
            center = np.asarray(self.geometry_center, dtype=float)
            if not np.isfinite(radius).all() or np.any(radius <= 0):
                raise ValueError('geometry_radius 必须为有限正数')
            if not np.isfinite(center).all():
                raise ValueError('geometry_center 必须有限')
        weights = np.asarray(self.penalty_weights, dtype=float)
        if (weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all()
                or np.any(weights < 0) or np.any(np.diff(weights) < 0)):
            raise ValueError('penalty_weights 必须是非空、有限、非负、非递减序列')
        object.__setattr__(self, 'penalty_weights', tuple(weights.tolist()))

    def to_dict(self):
        return _json(asdict(self))


@dataclass(frozen=True)
class InertialRecoveryResult:
    full_parameters: np.ndarray
    base_parameters: np.ndarray
    masses: np.ndarray
    centers_of_mass: np.ndarray
    inertias_com: np.ndarray
    extras: dict[str, float]
    parameter_names: tuple[str, ...]
    report: dict

    @property
    def success(self):
        return self.report['success']

    def to_dict(self):
        return {
            'success': self.success,
            'full_parameters': self.full_parameters.tolist(),
            'base_parameters': self.base_parameters.tolist(),
            'parameter_names': list(self.parameter_names),
            'masses': self.masses.tolist(),
            'centers_of_mass': self.centers_of_mass.tolist(),
            'inertias_com': self.inertias_com.tolist(),
            'extras': dict(self.extras),
            'links': [dict(link=i + 1, mass=float(m), center_of_mass=c.tolist(),
                           inertia_com=I.tolist())
                      for i, (m, c, I) in enumerate(zip(self.masses, self.centers_of_mass, self.inertias_com))],
            'units': {'mass': 'kg', 'center_of_mass': 'm', 'inertia_com': 'kg*m^2'},
            'inertia_frame': 'at center of mass, expressed in link coordinate axes',
            'report': _json(self.report),
        }


class _Layout:
    """物理 m/c/I 坐标到符号参数顺序的映射，两种符号排序都支持。"""
    def __init__(self, robot):
        s = robot.symbols
        self.n = s.dof
        self.names = tuple(str(p) for p in s.dynparms())
        pos = {name: i for i, name in enumerate(self.names)}
        self.mass = np.array([pos[str(p)] for p in s.m])
        self.first = np.array([[pos[str(p)] for p in h] for h in s.l])
        self.inertia = np.array([[pos[str(p)] for p in L] for L in s.Le])
        used = set(self.mass) | set(self.first.ravel()) | set(self.inertia.ravel())
        self.extra = np.array([i for i in range(len(self.names)) if i not in used], dtype=int)
        known_extras = {str(p): kind for kind in ('Ia', 'fv', 'fc', 'fo') for p in getattr(s, kind)}
        try:
            self.extra_kinds = [known_extras[self.names[i]] for i in self.extra]
        except KeyError as exc:
            raise ValueError(f'不支持的附加动力学参数: {exc}') from exc
        self.size = 10 * self.n + len(self.extra)

    def physical(self, x):
        n = self.n
        m = x[:n]
        c = x[n:4*n].reshape(n, 3)
        I = np.einsum('nk,kij->nij', x[4*n:10*n].reshape(n, 6), _TENSOR_BASIS)
        return m, c, I

    def full(self, x, *, jacobian=False):
        m, c, I = self.physical(x)
        Q = np.einsum('ni,ni->n', c, c)[:, None, None] * np.eye(3) - c[:, :, None] * c[:, None, :]
        L = I + m[:, None, None] * Q
        full = np.empty(len(self.names))
        full[self.mass] = m
        full[self.first] = m[:, None] * c
        full[self.inertia] = L[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
        full[self.extra] = x[10*self.n:]
        if not jacobian:
            return full
        J = np.zeros((len(self.names), self.size))
        for i in range(self.n):
            J[self.mass[i], i] = 1
            J[self.first[i], i] = c[i]
            J[self.first[i], self.n + 3*i + np.arange(3)] = m[i]
            J[self.inertia[i], i] = Q[i, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
            for k in range(3):
                e = np.eye(3)[k]
                dL = m[i] * (2*c[i, k]*np.eye(3) - np.outer(e, c[i]) - np.outer(c[i], e))
                J[self.inertia[i], self.n + 3*i + k] = dL[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
            J[self.inertia[i], 4*self.n + 6*i + np.arange(6)] = 1
        J[self.extra, 10*self.n + np.arange(len(self.extra))] = 1
        return full, J

    def from_full(self, full):
        full = np.asarray(full, dtype=float)
        if full.shape != (len(self.names),) or not np.isfinite(full).all():
            raise ValueError('nominal_parameters 必须是完整参数顺序的有限一维向量')
        m = full[self.mass]
        if np.any(m <= 0):
            raise ValueError('名义参数的质量必须为正')
        c = full[self.first] / m[:, None]
        L = np.einsum('nk,kij->nij', full[self.inertia], _TENSOR_BASIS)
        I = L - m[:, None, None] * (np.sum(c*c, axis=1)[:, None, None]*np.eye(3) - c[:, :, None]*c[:, None, :])
        return np.concatenate((m, c.ravel(), I[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)].ravel(), full[self.extra]))


def pack_inertial_parameters(robot, masses, centers_of_mass, inertias_com, *, extras=None):
    """把物理参数转成 ``robot.symbols.dynparms()`` 的顺序。

    惯量取质心处、在连杆坐标轴下表达，不是主轴系。
    没给的作动器 / 摩擦参数默认补零。这个转换本身不保证物理可行，
    可行性由恢复结果里的审计给出。
    """
    layout = _Layout(robot)
    m = _array(masses, (layout.n,), 'masses')
    c = _array(centers_of_mass, (layout.n, 3), 'centers_of_mass')
    I = _array(inertias_com, (layout.n, 3, 3), 'inertias_com')
    if np.any(m <= 0) or not np.allclose(I, I.transpose(0, 2, 1), rtol=1e-10, atol=1e-12):
        raise ValueError('质量必须为正，惯性张量必须对称')
    extra_names = [layout.names[i] for i in layout.extra]
    extras = {} if extras is None else {str(k): v for k, v in extras.items()}
    if set(extras) - set(extra_names):
        raise ValueError(f'未知附加参数: {sorted(set(extras) - set(extra_names))}')
    extra = _array([extras.get(name, 0.0) for name in extra_names], (len(extra_names),), 'extras')
    x = np.concatenate((m, c.ravel(), I[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)].ravel(), extra))
    return layout.full(x)


def _bounds(layout, config):
    n = layout.n
    ml = _array(config.mass_lower, (n,), 'mass_lower')
    mu = _array(config.mass_upper, (n,), 'mass_upper')
    cl = _array(config.com_lower, (n, 3), 'com_lower')
    cu = _array(config.com_upper, (n, 3), 'com_upper')
    if np.any(ml <= 0) or np.any(ml >= mu) or np.any(cl >= cu):
        raise ValueError('质量下界必须为正，质量/质心下界必须严格小于上界')
    if ((config.total_mass_lower is not None and config.total_mass_lower > mu.sum())
            or (config.total_mass_upper is not None and config.total_mass_upper < ml.sum())):
        raise ValueError('总质量范围与逐连杆质量上下界冲突')
    il = np.full((n, 6), -config.inertia_offdiagonal_bound)
    iu = np.full((n, 6), config.inertia_offdiagonal_bound)
    il[:, [0, 3, 5]] = config.inertia_diagonal_lower
    iu[:, [0, 3, 5]] = config.inertia_diagonal_upper
    el, eu = [], []
    for kind in layout.extra_kinds:
        el.append(-config.offset_bound if kind == 'fo' else 0.0)
        eu.append(config.offset_bound if kind == 'fo' else config.drive_inertia_upper if kind == 'Ia' else config.friction_upper)
    return (np.concatenate((ml, cl.ravel(), il.ravel(), el)),
            np.concatenate((mu, cu.ravel(), iu.ravel(), eu)))


def _geometry(layout, config):
    if config.geometry_radius is None:
        return None
    return (_array(config.geometry_center, (layout.n, 3), 'geometry_center'),
            _array(config.geometry_radius, (layout.n,), 'geometry_radius'))


def _geometry_violations(x, layout, config, *, gradient=False):
    """球包络的有符号无量纲违反量和解析 Jacobian；<=0 为可行。"""
    geometry = _geometry(layout, config)
    if geometry is None:
        value, jac = np.zeros(layout.n), np.zeros((layout.n, layout.size))
        return (value, jac) if gradient else value
    center, radius = geometry
    m, c, I = layout.physical(x)
    radius_sq = radius**2
    trace = np.trace(I, axis1=1, axis2=2)
    delta = c - center
    value = trace/(2*m*radius_sq) + np.sum(delta*delta, axis=1)/radius_sq - 1
    if not gradient:
        return value
    jac = np.zeros((layout.n, layout.size))
    for i in range(layout.n):
        jac[i, i] = -trace[i]/(2*m[i]**2*radius_sq[i])
        jac[i, layout.n + 3*i:layout.n + 3*i + 3] = 2*delta[i]/radius_sq[i]
        jac[i, 4*layout.n + 6*i + np.array([0, 3, 5])] = 1/(2*m[i]*radius_sq[i])
    return value, jac


def _prior_scales(initial, layout, config, lower, upper):
    """先验尺度只由名义参数和配置确定，不随待优化参数变化。"""
    m, _, I = layout.physical(initial)
    geometry = _geometry(layout, config)
    length = (geometry[1] if geometry is not None
              else np.linalg.norm((upper[layout.n:4*layout.n]
                                   - lower[layout.n:4*layout.n]).reshape(layout.n, 3)/2, axis=1))
    inertia = np.maximum(np.linalg.norm(I, axis=(1, 2)), 1e-6*m*length**2)
    return dict(mass=m.copy(), center_of_mass=length.copy(), inertia_com=inertia)


def _nominal_prior(x, initial, layout, scales, *, gradient=False):
    """每连杆的相对质量、质心位移、Frobenius 惯量偏差平方均值。"""
    m, c, I = layout.physical(x)
    m0, c0, I0 = layout.physical(initial)
    dm, dc, dI = m - m0, c - c0, I - I0
    ms, cs, Is = scales['mass'], scales['center_of_mass'], scales['inertia_com']
    denominator = 3*layout.n
    value = float((np.sum((dm/ms)**2) + np.sum((dc/cs[:, None])**2)
                   + np.sum((dI/Is[:, None, None])**2))/denominator)
    if not gradient:
        return value
    grad = np.zeros(layout.size)
    grad[:layout.n] = 2*dm/ms**2/denominator
    grad[layout.n:4*layout.n] = (2*dc/cs[:, None]**2/denominator).ravel()
    grad[4*layout.n:10*layout.n] = (2*np.einsum('nij,kij->nk', dI, _TENSOR_BASIS)
                                                /Is[:, None]**2/denominator).ravel()
    return value, grad


def _physical_penalty(x, layout, config, *, gradient=False):
    """参考算法的线性 hinge 罚项（不是平方）及其次梯度。"""
    m, _, I = layout.physical(x)
    penalty = 0.0
    grad = np.zeros(layout.size)
    total = m.sum()
    if config.total_mass_lower is not None and total < config.total_mass_lower:
        penalty += config.total_mass_lower - total
        grad[:layout.n] -= 1
    if config.total_mass_upper is not None and total > config.total_mass_upper:
        penalty += total - config.total_mass_upper
        grad[:layout.n] += 1
    values, vectors = np.linalg.eigh(I)
    violations = values[:, -1] - 0.5 * np.trace(I, axis1=1, axis2=2)
    for i in np.flatnonzero(violations > 0):
        penalty += violations[i]
        v = vectors[i, :, -1]
        derivative = np.outer(v, v) - 0.5*np.eye(3)
        grad[4*layout.n + 6*i:4*layout.n + 6*i + 6] = np.einsum('ij,kij->k', derivative, _TENSOR_BASIS)
    if config.geometry_radius is not None:
        violations, jac = _geometry_violations(x, layout, config, gradient=True)
        active = violations > 0
        penalty += np.maximum(violations, 0).sum()
        grad += jac[active].sum(axis=0)
    return (float(penalty), grad) if gradient else float(penalty)


def _audit(x, layout, config, lower, upper):
    m, c, I = layout.physical(x)
    eig = np.linalg.eigvalsh(I)
    margins = np.trace(I, axis1=1, axis2=2)/2 - eig[:, -1]
    violation = np.maximum(np.maximum(lower-x, x-upper), 0)
    total_violation = max(0., (config.total_mass_lower or 0.) - m.sum(),
                          m.sum() - (config.total_mass_upper if config.total_mass_upper is not None else np.inf))
    tol = config.feasibility_tolerance
    geometry = _geometry(layout, config)
    geometry_signed = _geometry_violations(x, layout, config)
    geometry_violations = np.maximum(geometry_signed, 0)
    geometry_feasible = bool(np.isfinite(geometry_signed).all() and geometry_violations.max() <= tol)
    feasible = bool(np.isfinite(x).all() and np.all(m > 0) and violation.max() <= tol
                    and total_violation <= tol and eig.min() >= -tol and margins.min() >= -tol
                    and geometry_feasible)
    # 报告接近界的位置，而不把贴界本身视为物理失败。
    contact_tol = np.maximum(tol, 1e-4*(upper-lower))
    mass_contacts, com_contacts, inertia_contacts = [], [], []
    for bound_name, bounds in (('lower', lower), ('upper', upper)):
        near = np.abs(x-bounds) <= contact_tol
        for i in np.flatnonzero(near[:layout.n]):
            mass_contacts.append(dict(link=int(i+1), bound=bound_name,
                                      value=float(x[i]), bound_value=float(bounds[i])))
        for j in np.flatnonzero(near[layout.n:4*layout.n]):
            index = layout.n + j
            com_contacts.append(dict(link=int(j//3+1), component=('x', 'y', 'z')[j%3],
                                     bound=bound_name, value=float(x[index]), bound_value=float(bounds[index])))
        for j in np.flatnonzero(near[4*layout.n:10*layout.n]):
            index = 4*layout.n + j
            inertia_contacts.append(dict(link=int(j//6+1), component=('xx', 'xy', 'xz', 'yy', 'yz', 'zz')[j%6],
                                         bound=bound_name, value=float(x[index]), bound_value=float(bounds[index])))
    return dict(physical_feasible=feasible, total_mass=float(m.sum()),
                max_bound_violation=float(violation.max()), total_mass_violation=float(total_violation),
                inertia_eigenvalues=eig.tolist(), triangle_margins=margins.tolist(),
                geometry_enabled=geometry is not None, geometry_feasible=geometry_feasible,
                geometry_signed_violations=geometry_signed.tolist(),
                geometry_violations=geometry_violations.tolist(),
                geometry_violation_definition='max(trace(I_com)/(2*m*r^2) + ||c-center||^2/r^2 - 1, 0)',
                geometry_constraint='selected spherical mass-distribution envelope; second-moment constraint, not exact CAD geometry',
                geometry_second_moment_margins=(None if geometry is None
                    else (-geometry_signed*m*geometry[1]**2).tolist()),
                mass_bound_contacts=mass_contacts, com_bound_contacts=com_contacts,
                inertia_bound_contacts=inertia_contacts,
                bound_contact_definition='distance <= max(feasibility_tolerance, 1e-4*(upper-lower))',
                physical_penalty=_physical_penalty(x, layout, config))


def recover_inertial_parameters(robot, base_parameters, *, nominal_parameters=None,
                                 config=None, progress=None):
    """基参数辨识之后，独立于仿真恢复 m/质心/惯量。

    先调用 ``robot.dynamics.calc_base_parms()``。``base_parameters`` 必须
    用同一组基。``nominal_parameters`` 初始化第 0 次运行；开启
    ``nominal_prior_weight`` 后同时用作 m/c/I 软先验，其余运行随机起步。
    目标为 ||B*pi-target||^2 + max(||target||,1)^2 * prior_weight * prior
    + penalty_weight * physical_penalty。附加作动器参数不参与先验。
    最终优先选择求解器收敛、物理可行且基参数误差通过的候选，再比较损失。
    先验主要约束不可辨识或弱约束方向，并允许拟合折中；不证明逐连杆参数等于真值。
    """
    try:
        from scipy.optimize import dual_annealing, minimize
    except ImportError as exc:
        raise ImportError('惯性参数恢复需要 scipy，请安装 robot-model[ident]') from exc
    config = RecoveryConfig() if config is None else config
    if not isinstance(config, RecoveryConfig):
        raise TypeError('config 必须是 RecoveryConfig')
    layout = _Layout(robot)
    dyn = robot.dynamics
    if any(getattr(dyn, name, None) is None for name in ('Pb', 'Pd', 'Kd')):
        raise ValueError('请先调用 robot.dynamics.calc_base_parms() 确定基参数映射')
    B = np.asarray(dyn.Pb, float).T + np.asarray(dyn.Kd, float) @ np.asarray(dyn.Pd, float).T
    target = np.asarray(base_parameters, dtype=float)
    if (B.ndim != 2 or B.shape[1] != len(layout.names) or B.shape[0] == 0
            or target.shape != (B.shape[0],) or not np.isfinite(B).all() or not np.isfinite(target).all()):
        raise ValueError('基参数/映射维度不匹配，或包含非有限值')
    lower, upper = _bounds(layout, config)
    span = upper - lower
    initial = None if nominal_parameters is None else layout.from_full(nominal_parameters)
    if config.nominal_prior_weight > 0 and initial is None:
        raise ValueError('nominal_prior_weight > 0 时必须提供 nominal_parameters')
    _geometry(layout, config)  # 在调用优化器前验证可广播维度。
    scales = None if initial is None else _prior_scales(initial, layout, config, lower, upper)
    rng = np.random.default_rng(config.seed)
    history, candidates = [], []
    normalization = max(float(np.linalg.norm(target)), 1.0)
    prior_factor = normalization**2*config.nominal_prior_weight

    for run in range(config.runs):
        x = (np.clip(initial, lower, upper) if run == 0 and initial is not None
             else rng.uniform(lower, upper))
        z = (x - lower)/span
        for stage, weight in enumerate(config.penalty_weights):
            def objective(z, with_grad=False):
                x = lower + span*z
                if with_grad:
                    full, J = layout.full(x, jacobian=True)
                    penalty, grad = _physical_penalty(x, layout, config, gradient=True)
                    residual = B @ full - target
                    value = float(residual @ residual + weight*penalty)
                    derivative = 2*(B @ J).T @ residual + weight*grad
                    if prior_factor > 0:
                        prior, prior_grad = _nominal_prior(x, initial, layout, scales, gradient=True)
                        value += prior_factor*prior
                        derivative += prior_factor*prior_grad
                    return value, derivative*span
                residual = B @ layout.full(x) - target
                value = float(residual @ residual + weight*_physical_penalty(x, layout, config))
                if prior_factor > 0:
                    value += prior_factor*_nominal_prior(x, initial, layout, scales)
                return value

            annealed = dual_annealing(objective, [(0., 1.)]*layout.size, x0=z,
                                     maxiter=config.annealing_maxiter, seed=rng, no_local_search=True)
            local = minimize(lambda v: objective(v, True), annealed.x, jac=True, method='SLSQP',
                             bounds=[(0., 1.)]*layout.size,
                             options={'maxiter': config.local_maxiter, 'ftol': config.ftol})
            # 局部精修失败也不能丢掉退火的最好点。
            if np.isfinite(local.x).all() and objective(local.x) <= objective(annealed.x):
                z, converged, message = local.x, bool(local.success), str(local.message)
            else:
                z, converged, message = annealed.x, False, '局部优化未改进；保留退火候选。' + str(local.message)
            x = lower + span*z
            residual = B @ layout.full(x) - target
            audit = _audit(x, layout, config, lower, upper)
            record = dict(run=run, stage=stage, penalty_weight=float(weight),
                          initialization='nominal' if run == 0 and initial is not None else 'random',
                          loss=objective(z), base_residual_norm=float(np.linalg.norm(residual)),
                          physical_penalty=audit['physical_penalty'],
                          physical_feasible=audit['physical_feasible'], solver_success=converged,
                          geometry_feasible=audit['geometry_feasible'],
                          nominal_prior_penalty=(None if initial is None
                              else _nominal_prior(x, initial, layout, scales)),
                          local_iterations=int(local.nit), message=message)
            history.append(record)
            if progress is not None:
                progress(dict(record))
        accepted = bool(converged and audit['physical_feasible']
                        and record['base_residual_norm']/normalization <= config.max_base_error)
        candidates.append((record['loss'], x.copy(), converged, message, run, accepted))

    loss, x, converged, message, selected_run, _ = min(candidates, key=lambda item: (not item[5], item[0]))
    full = layout.full(x)
    reconstructed = B @ full
    residual = reconstructed - target
    error = float(np.linalg.norm(residual)/normalization)
    audit = _audit(x, layout, config, lower, upper)
    matches = error <= config.max_base_error
    success = bool(converged and audit['physical_feasible'] and matches)
    m, c, I = layout.physical(x)
    prior = None if initial is None else _nominal_prior(x, initial, layout, scales)
    report = dict(success=success, solver_success=converged, **audit, matches_base=bool(matches),
                  base_error=error, base_residual_norm=float(np.linalg.norm(residual)),
                  base_residual=residual.tolist(), target_base_parameters=target.tolist(),
                  base_error_definition='norm(B*pi - target) / max(norm(target), 1)',
                  iterations=sum(h['local_iterations'] for h in history),
                  message=message, selected_run=selected_run, loss=float(loss),
                  candidate_selection='successful candidates first, then minimum penalized loss',
                  nominal_prior_enabled=bool(config.nominal_prior_weight > 0),
                  nominal_prior_weight=float(config.nominal_prior_weight),
                  nominal_prior_penalty=prior,
                  nominal_prior_rms=None if prior is None else float(np.sqrt(prior)),
                  nominal_prior_loss=0.0 if prior is None else float(prior_factor*prior),
                  nominal_prior_scales=None if scales is None else _json(scales),
                  nominal_prior_definition='sum_links((dm/m0)^2 + ||dc/length||^2 + ||dI/inertia_scale||_F^2)/(3*n); extras excluded',
                  nominal_prior_scale_definition='mass=m0; length=geometry_radius or COM-box half-diagonal; inertia_scale=max(||I0||_F, 1e-6*m0*length^2)',
                  objective_definition='||B*pi-target||^2 + max(||target||,1)^2*nominal_prior_weight*nominal_prior_penalty + penalty_weight*physical_penalty',
                  config=config.to_dict(), attempts=history,
                  method='two-step penalty continuation: dual_annealing + SLSQP',
                  initial_source='nominal first run, random subsequent runs' if initial is not None else 'random bounds',
                  nominal_initial_clipped=bool(initial is not None and np.any((initial < lower) | (initial > upper))),
                  uniqueness='one feasible parameter set; individual link parameters are generally not uniquely identifiable')
    return InertialRecoveryResult(full, reconstructed, m.copy(), c.copy(), I.copy(),
                                  {layout.names[i]: float(full[i]) for i in layout.extra}, layout.names, report)
