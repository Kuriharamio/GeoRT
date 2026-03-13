import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

os.environ['DISCOVERSE_ASSETS_DIR'] = os.path.join(PROJECT_ROOT, 'models')

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
import mujoco
import numpy as np

class SimBaseCfg(BaseConfig):
    mjcf_file_path = "mjcf/simple_env.xml"
    decimation     = 4
    timestep       = 0.002
    sync           = True
    headless       = True
    render_set     = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080,
    }
    obs_rgb_cam_id = None
    obs_depth_cam_id = None
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict = {}
    rb_link_list = list(gs_model_dict.keys())
    
    enable_viewer = True

class SimBase(SimulatorBase):

    def __init__(self, config: SimBaseCfg):
        self.enable_viewer = getattr(config, "enable_viewer", True)

        super().__init__(config)
        print("="*80)

        # render set
        self.render_fps = 120
        self.time_step = 1./self.render_fps
        self.render_gap = int(1.0 / self.render_fps / self.mj_model.opt.timestep)
        self.cross_speed = 1.0

        # viewer
        self._init_viewer()

    ##################################
    # Simulation Method
    ##################################

    def resetState(self):
        super().resetState()

    def step(self):
        super().step()

    def post_physics_step(self):
        self._sync_viewer()

    def updateControl(self, action):
        self.mj_data.ctrl[:] = np.clip(self.mj_data.ctrl[:], self.mj_model.actuator_ctrlrange[:,0], self.mj_model.actuator_ctrlrange[:,1])

    def post_load_mjcf(self):
        self.mj_model.vis.map.znear = 0.00001

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "img"  : self.img_rgb_obs_s.copy(),
            "depth" : self.img_depth_obs_s.copy()
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None
    
    def checkTerminated(self):
        return False

    ##################################
    # Viewer
    ##################################

    def _init_viewer(self):
        """初始化查看器"""
        self.mj_viewer = None
        self.last_viewer_select = -1
        self.measure_only_mode = True
        self.last_sim_time = 0.0
        self._executing_init = False
        self._last_approach_state = "idle"

        if not self.enable_viewer:
            print("[INFO] mujoco.viewer disabled.")
            return

        self.last_selected_body_name = None
        self.last_selected_body_pos = None

        try:
            import mujoco.viewer as mjv
            self.mj_viewer = mjv.launch_passive(
                self.mj_model,
                self.mj_data,
                show_left_ui=False,
                show_right_ui=False,
            )
            if self.mj_viewer:
                self.mj_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.mj_viewer.cam.lookat[:] = [0.1, -0.001, 0.0]
                self.mj_viewer.cam.distance = 0.3
                self.mj_viewer.cam.azimuth = 225
                self.mj_viewer.cam.elevation = -20
            print("[INFO] 已初始化 mujoco viewer")
        except Exception as e:
            print(f"[WARN] 查看器启动失败: {e}")
            self.mj_viewer = None

    def _sync_viewer(self):
        """同步查看器"""
        if self.mj_viewer is None:
            self._update_tactile_sensor_manager()
            return
        try:
            current_time = self.mj_data.time
            if not self._executing_init and current_time < self.last_sim_time - 0.01:
                print("[Viewer] 检测到仿真重置")
                self.resetState()
            self.last_sim_time = current_time
            self.mj_viewer.sync()
        except Exception as e:
            print(f"[WARN] 查看器同步失败: {e}")

    ##################################
    # Cleanup Methods
    ##################################

    def _cleanup_before_exit(self):
        """清理所有资源，防止内存泄漏"""
        if hasattr(self, 'tactile_sensor_manager') and self.tactile_sensor_manager is not None:
            try:
                self.tactile_sensor_manager.close()
            except Exception:
                pass

        # 清理查看器
        if self.mj_viewer is not None:
            try:
                self.mj_viewer.close()
            except Exception as e:
                print(f"[WARN] 关闭查看器时出错: {e}")
            self.mj_viewer = None

        super()._cleanup_before_exit()

##################################
# Main Function
##################################

if __name__ == "__main__":
    cfg = SimBaseCfg()
    cfg.obs_rgb_cam_id = None

    sim = SimBase(cfg)
    sim.reset()

    try:
        while sim.running and (sim.mj_viewer is None or sim.mj_viewer.is_running()):
            sim.step()
    except KeyboardInterrupt:
        print("[INFO] Ctrl C 关闭")
    except Exception as e:
        print(e)
    finally:
        sim._cleanup_before_exit()
        print("[INFO] Simulation ended.")
