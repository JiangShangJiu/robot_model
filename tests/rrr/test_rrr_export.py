"""导出的三个后端：生成物必须能跑，且跑出来和 MuJoCo 真值一致。

只检查"生成了文本"没有意义——代码生成的坑都在 cse 变量、
参数下标顺序、打印器这些地方，所以这里一律执行生成物再比数值。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess

import numpy as np
import pytest

from robot_model import Robot
from robot_model.export import AVAILABLE_KEYS
from robot_model.simulation.reference import MujocoReference
from robot_model.robots.rrr import (
    RRR_BODY_NAMES,
    RRR_MDH_PARMS,
    rrr_xml_path,
)

mujoco = pytest.importorskip("mujoco")

ATOL = 1e-9


@pytest.fixture(scope="module")
def robot():
    return Robot.from_mdh("rrr_arm", RRR_MDH_PARMS)


@pytest.fixture(scope="module")
def ref():
    model = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
    return MujocoReference(model, RRR_BODY_NAMES)


@pytest.fixture(scope="module")
def generated(robot, tmp_path_factory):
    """生成并 import 一次，多个用例共用。"""
    path = tmp_path_factory.mktemp("export") / "rrr_model.py"
    robot.export.python(path, keys=["T", "M", "c", "g", "tau", "H"])
    spec = importlib.util.spec_from_file_location("rrr_generated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _states():
    rng = np.random.default_rng(11)
    return [
        (
            rng.uniform(-1.5, 1.5, 3),
            rng.uniform(-1, 1, 3),
            rng.uniform(-2, 2, 3),
        )
        for _ in range(4)
    ]


def test_generated_python_matches_mujoco(generated, ref):
    pi = ref.dynparms()
    for q, dq, ddq in _states():
        tau_mj = ref.inverse_dynamics(q, dq, ddq)
        assert np.allclose(generated.T(q), ref.body_poses(q)[-1], atol=ATOL)
        assert np.allclose(generated.tau(q, dq, ddq, pi), tau_mj, atol=ATOL)
        assert np.allclose(generated.H(q, dq, ddq) @ pi, tau_mj, atol=ATOL)
        split = (
            generated.M(q, pi) @ ddq
            + generated.c(q, dq, pi)
            + generated.g(q, pi)
        )
        assert np.allclose(split, tau_mj, atol=ATOL)


def test_column_vectors_name_each_component(robot):
    """tau 单个分量上千字符，必须逐关节命名后再组装，否则 return 读不了。"""
    text = robot.export.python(keys=["tau"])
    assert "    tau_1 = " in text
    assert "    tau_3 = " in text
    assert "return numpy.array([tau_1, tau_2, tau_3])" in text
    # 矩阵仍按行输出，逐元素命名只会变成噪声
    assert "M_1" not in robot.export.python(keys=["M"])


def test_generated_python_shapes(generated, ref):
    q, dq, ddq = _states()[0]
    pi = ref.dynparms()
    assert generated.T(q).shape == (4, 4)
    assert generated.M(q, pi).shape == (3, 3)
    assert generated.tau(q, dq, ddq, pi).shape == (3,)
    assert generated.H(q, dq, ddq).shape == (3, len(pi))


@pytest.mark.skipif(shutil.which("gcc") is None, reason="需要 gcc")
def test_generated_c_matches_python(robot, ref, generated, tmp_path):
    src = tmp_path / "rrr_model.c"
    robot.export.c(src, keys=["tau"])

    q, dq, ddq = _states()[0]
    pi = ref.dynparms()

    def carray(name, v):
        body = ", ".join(repr(float(x)) for x in v)
        return f"    const double {name}[] = {{{body}}};\n"

    main = (
        "#include <stdio.h>\n"
        "void tau(const double *q, const double *dq, const double *ddq,\n"
        "         const double *parms, double *out);\n"
        "int main(void) {\n"
        + carray("q", q)
        + carray("dq", dq)
        + carray("ddq", ddq)
        + carray("parms", pi)
        + "    double out[3];\n"
        "    tau(q, dq, ddq, parms, out);\n"
        '    for (int i = 0; i < 3; ++i) printf("%.17g\\n", out[i]);\n'
        "    return 0;\n}\n"
    )
    (tmp_path / "main.c").write_text(main)

    exe = tmp_path / "rrr_test"
    build = subprocess.run(
        ["gcc", "-O2", "-o", str(exe), str(src), str(tmp_path / "main.c"), "-lm"],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    tau_c = np.array([float(x) for x in run.stdout.split()])
    assert np.allclose(tau_c, ref.inverse_dynamics(q, dq, ddq), atol=ATOL)
    assert np.allclose(tau_c, generated.tau(q, dq, ddq, pi), atol=ATOL)


def test_latex_covers_requested_items(robot):
    tex = robot.export.latex(keys=["T", "g"], standalone=True)
    assert "\\documentclass" in tex and "\\end{document}" in tex
    assert "末端位姿" in tex and "重力项" in tex
    frag = robot.export.latex(keys=["T"], standalone=False)
    assert "\\documentclass" not in frag


def test_latex_cse_style_emits_temporaries(robot):
    """大表达式必须走 cse 排版，否则一行公式能撑爆页面。"""
    tex = robot.export.latex(keys=["tau"], standalone=False)
    assert "中间量" in tex


def test_latex_lines_stay_within_page(robot):
    """没有折行 / 提取中间量的话，tau 会有 400 字符的单行冲出页面。"""
    tex = robot.export.latex(keys=["tau", "M", "c"], standalone=False)
    too_long = [ln for ln in tex.splitlines() if len(ln) > 200]
    assert not too_long, f"{len(too_long)} 行过长，最长 {max(map(len, too_long))}"


def test_latex_escapes_underscore_in_title(robot):
    """机器人名常带下划线，不转义会让 pdflatex 报错或渲染成下标。"""
    tex = robot.export.latex(keys=["T"], standalone=True)
    assert "\\title{rrr\\_arm}" in tex


def test_per_link_keys(robot):
    items = robot.export.items(["T_1", "J_2", "T_rel_3"])
    assert [it.key for it in items] == ["T_1", "J_2", "T_rel_3"]
    assert items[0].expr.shape == (4, 4)
    assert items[1].expr.shape == (6, 3)


def test_unknown_key_rejected(robot):
    with pytest.raises(KeyError):
        robot.export.items(["nope"])
    with pytest.raises(ValueError):
        robot.export.items(["T_9"])  # 超出自由度


def test_available_keys_all_build(robot):
    """每个公开 key 都要真能取出表达式，避免文档与实现脱节。"""
    for key in AVAILABLE_KEYS:
        (item,) = robot.export.items([key], simplify=False)
        assert item.expr.free_symbols or item.expr.is_zero_matrix
