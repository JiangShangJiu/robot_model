"""驱动器侧附加项：关节摩擦与驱动（转子）惯量。"""

from __future__ import annotations

from typing import Callable

from sympy import sign, zeros

from ..utils.mathutil import identity
from ..utils.symbols import RobotSymbols

_FRICTION_TERMS = {"Coulomb", "viscous", "offset"}


def friction_force(symbols: RobotSymbols, ifunc: Callable | None = None):
    ifunc = ifunc or identity
    fric = zeros(symbols.dof, 1)
    if not symbols.frictionmodel:
        return fric

    asked = set(symbols.frictionmodel)
    unknown = asked - _FRICTION_TERMS
    if unknown:
        raise ValueError(
            f"unknown friction terms {unknown}; use subset of {_FRICTION_TERMS}"
        )

    for i in range(symbols.dof):
        if "viscous" in asked:
            fric[i] += symbols.fv[i] * symbols.dq[i]
        if "Coulomb" in asked:
            fric[i] += symbols.fc[i] * sign(symbols.dq[i])
        if "offset" in asked:
            fric[i] += symbols.fo[i]
        fric[i] = ifunc(fric[i])
    return fric


def drive_inertia_term(symbols: RobotSymbols, ifunc: Callable | None = None):
    ifunc = ifunc or identity
    out = zeros(symbols.dof, 1)
    if symbols.driveinertiamodel is None:
        return out
    if symbols.driveinertiamodel != "simplified":
        raise ValueError(
            f"unknown driveinertiamodel {symbols.driveinertiamodel!r}; "
            "use None or 'simplified'"
        )
    for i in range(symbols.dof):
        out[i] = ifunc(symbols.Ia[i] * symbols.ddq[i])
    return out
