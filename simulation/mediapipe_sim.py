import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.base_env import SimBase, SimBaseCfg
from discoverse.utils import get_control_idx
from geort.mocap.mediapipe_mocap import MediaPipeMocap
from geort import load_model, get_config

class SimMediaPipeCfg(SimBaseCfg):
    hand_name = "discoverdex"   # 手型，对应 geort/config 下的配置（如 discoverdex.json）
    ckpt_tag = "tag"           # GeoRT checkpoint 标识，用于 load_model(ckpt_tag)


class SimMediaPipe(SimBase):

    def __init__(self, config: SimMediaPipeCfg):
        self._hand_name = getattr(config, "hand_name", "discoverdex")
        self._ckpt_tag = getattr(config, "ckpt_tag", "last")

        super().__init__(config)

        self._mocap = MediaPipeMocap()
        self._model = load_model(self._ckpt_tag)
        self._hand_config = get_config(self._hand_name)
        joint_order = self._hand_config["joint_order"]
        self._n_hand_dof = len(joint_order)

        control_idx = get_control_idx(self.mj_model, joint_order, check=True)
        self._qpos_to_ctrl = [control_idx[name] for name in joint_order]
        self._last_qpos = None
        self._n_ctrl = self.mj_model.nu
        print(
            f"[SimMediaPipe] hand={self._hand_name} ckpt={self._ckpt_tag} "
            f"n_hand_dof={self._n_hand_dof} n_ctrl={self._n_ctrl} "
            f"qpos_to_ctrl={[c for c in self._qpos_to_ctrl]}"
        )

    def get_mocap_qpos(self):
        result = self._mocap.get()
        if result["status"] == "recording" and result.get("result") is not None:
            keypoints = np.asarray(result["result"], dtype=np.float64)
            if keypoints.size >= 21 * 3:
                qpos = self._model.forward(keypoints)
                qpos = np.asarray(qpos).flatten()
                if len(qpos) == self._n_hand_dof:
                    self._last_qpos = qpos
                    return qpos
        if self._last_qpos is not None:
            return self._last_qpos
        return None

    def updateControl(self, action):
        if action is None:
            return
        action = np.asarray(action).flatten()
        low = self.mj_model.actuator_ctrlrange[:, 0]
        high = self.mj_model.actuator_ctrlrange[:, 1]
        for i, ctrl_idx in enumerate(self._qpos_to_ctrl):
            if ctrl_idx is not None and i < len(action):
                val = np.clip(float(action[i]), low[ctrl_idx], high[ctrl_idx])
                self.mj_data.ctrl[ctrl_idx] = val

    def step(self):
        qpos = self.get_mocap_qpos()
        self.updateControl(qpos)
        super().step()


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe 动捕 + GeoRT 关节映射驱动 MuJoCo 仿真"
    )
    parser.add_argument("-hand", type=str, default="discoverdex", help="手型配置名（如 discoverdex）")
    parser.add_argument("-ckpt_tag", type=str, default="last", help="GeoRT 模型 checkpoint 标识")
    parser.add_argument("--no-viewer", action="store_true", help="禁用 viewer")
    args = parser.parse_args()

    cfg = SimMediaPipeCfg()
    cfg.hand_name = args.hand
    cfg.ckpt_tag = args.ckpt_tag
    cfg.enable_viewer = not args.no_viewer

    sim = SimMediaPipe(cfg)
    sim.reset()

    try:
        while sim.running and (sim.mj_viewer is None or sim.mj_viewer.is_running()):
            sim.step()
    except KeyboardInterrupt:
        print("[INFO] Ctrl+C 退出")
    except Exception as e:
        print(e)
    finally:
        sim._cleanup_before_exit()
        print("[INFO] Simulation ended.")


if __name__ == "__main__":
    main()
