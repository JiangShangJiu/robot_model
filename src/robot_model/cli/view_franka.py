"""用 MuJoCo 可视化 Franka Panda，并可与 robot_model FK 对照。

用法（先在仓库根目录 pip install -e ".[viz]"）::

    python -m robot_model.cli.view_franka
    python -m robot_model.cli.view_franka --q 0,0,0,-1.57,0,1.57,-0.785
    python -m robot_model.cli.view_franka --home

窗口里可用鼠标旋转/缩放。终端会打印 link7 位姿误差（相对本库 FK）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


from robot_model import Robot
from robot_model.robots.franka import PANDA_MDH_PARMS, panda_xml_path

if TYPE_CHECKING:
    import mujoco


def _scene_xml() -> Path:
    return panda_xml_path().parent / "_viz_q0_scene.xml"


def _parse_q(text: str | None, model: mujoco.MjModel, *, home: bool) -> np.ndarray:
    narm = 7
    if home:
        return np.array(model.key("home").qpos[:narm], dtype=float)
    if text is None:
        return np.zeros(narm, dtype=float)
    q = np.asarray([float(x) for x in text.split(",")], dtype=float)
    if q.size != narm:
        raise SystemExit(f"--q 需要 {narm} 个数，当前 {q.size}")
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description="MuJoCo 可视化 Franka + robot_model FK")
    ap.add_argument(
        "--q",
        default=None,
        help="7 个关节角（弧度），逗号分隔；默认全 0",
    )
    ap.add_argument("--home", action="store_true", help="用模型 keyframe home")
    ap.add_argument(
        "--xml",
        default=None,
        help="MJCF 路径；默认 assets/.../_viz_q0_scene.xml",
    )
    args = ap.parse_args()
    import mujoco
    import mujoco.viewer


    xml = Path(args.xml) if args.xml else _scene_xml()
    if not xml.is_file():
        xml = panda_xml_path()
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    q = _parse_q(args.q, model, home=args.home)
    data.qpos[:] = 0.0
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)

    robot = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
    T = robot.fk_numpy(q)
    link7 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    pos_err = float(np.linalg.norm(T[:3, 3] - data.xpos[link7]))
    rot_err = float(
        np.linalg.norm(T[:3, :3] @ data.xmat[link7].reshape(3, 3).T - np.eye(3))
    )

    print(f"xml: {xml}")
    print(f"q:   {np.array2string(q, precision=4)}")
    print(f"FK link7 pos_err={pos_err:.3e}  rot_err={rot_err:.3e}")
    print("打开 MuJoCo 窗口；关闭窗口后退出。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 运动学位姿：不步进物理，只保持当前 q
            data.qpos[:7] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
