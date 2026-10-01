"""迁移后公开导入在独立进程、任意工作目录及无仿真依赖时仍可用。"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["data", "excitation", "simulation", "identification"])
def test_import_order_and_optional_dependencies(first, tmp_path):
    src = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'scipy'}:
            raise AssertionError('optional dependency imported eagerly: ' + fullname)

sys.meta_path.insert(0, BlockOptional())
importlib.import_module('robot_model.' + sys.argv[1])
from robot_model import Robot
from robot_model.data import MotionData
from robot_model.excitation import FourierTrajectory, screen_excitation, search_excitation
from robot_model.simulation import RealisticMujocoSource
from robot_model import identification
from robot_model.simulation.tracking import _require_mujoco
from robot_model.experiments.rrr import run_experiment
from robot_model.experiments import prepare_measurements
from robot_model.robots import rrr_xml_path, panda_xml_path
assert hasattr(identification, 'identify')
assert not hasattr(identification, 'MotionData')
assert not hasattr(identification, 'FourierTrajectory')
assert callable(_require_mujoco)
assert callable(run_experiment)
assert callable(prepare_measurements)
assert rrr_xml_path().is_file()
assert panda_xml_path().is_file()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(src)), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
