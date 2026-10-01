"""通用 CLI 的机器人默认值和显式覆盖不应依赖昂贵的仿真实验。"""

import sys

import pytest

from robot_model.cli import identify_robot


@pytest.mark.parametrize("robot,arguments,expected", [
    ("rrr", [], True),
    ("franka", [], False),
    ("rrr", ["--optimize-excitation"], True),
    ("franka", ["--optimize-excitation"], True),
    ("rrr", ["--no-optimize-excitation"], False),
    ("franka", ["--no-optimize-excitation"], False),
])
def test_optimization_defaults_and_explicit_overrides(monkeypatch, robot, arguments, expected):
    config = object()
    captured = {}

    def run_experiment(actual_config, **kwargs):
        assert actual_config is config
        captured.update(kwargs)
        return {"status": "completed"}

    def load_robot(name):
        assert name == robot
        return lambda: config, run_experiment

    monkeypatch.setattr(identify_robot, "_load_robot", load_robot)
    monkeypatch.setattr(sys, "argv", ["robot-identify", "--robot", robot, *arguments])
    identify_robot.main()
    assert captured["optimize"] is expected


def test_help_explains_robot_specific_optimization_defaults(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["robot-identify", "--help"])
    with pytest.raises(SystemExit) as caught:
        identify_robot.main()
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "RRR 默认开启，Franka 默认关闭" in help_text
    assert "--optimize-excitation" in help_text
    assert "--no-optimize-excitation" in help_text
