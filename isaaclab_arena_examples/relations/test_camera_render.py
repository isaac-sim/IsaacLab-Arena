# Minimal test: does the Camera sensor produce non-black images in this container?
# Usage: /isaac-sim/python.sh isaaclab_arena_examples/relations/test_camera_render.py --enable_cameras --headless

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg

sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
sim = sim_utils.SimulationContext(sim_cfg)

# Ground + light + colored cube
sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DistantLightCfg(intensity=3000.0))

cube_cfg = RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
)
cube = RigidObject(cfg=cube_cfg)

# Camera
sim_utils.create_prim("/World/CameraMount", "Xform")
camera = Camera(cfg=CameraCfg(
    prim_path="/World/CameraMount/CameraSensor",
    update_period=0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
))

sim.reset()
camera.update(dt=0.0)

camera.set_world_poses_from_view(
    torch.tensor([[2.0, 2.0, 2.0]], device="cuda:0"),
    torch.tensor([[0.0, 0.0, 0.0]], device="cuda:0"),
)

for i in range(20):
    sim.step()
    camera.update(dt=sim.get_physics_dt())

rgb = camera.data.output["rgb"]
img = rgb[0].cpu().numpy().astype(np.uint8)
pixel_sum = int(img.sum())
print(f"Frame shape: {img.shape}, sum: {pixel_sum}")

if pixel_sum > 0:
    from PIL import Image
    Image.fromarray(img[:, :, :3]).save("/tmp/test_render.png")
    print("SUCCESS: Saved /tmp/test_render.png")
else:
    print("FAIL: Image is all black")

simulation_app.close()
