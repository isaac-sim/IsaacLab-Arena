import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm
path_prefix = '/datasets/curobo_v3_with_camera.hdf5'
# Load the HDF5 file
file = h5py.File(path_prefix, 'r')

for i in tqdm(range(1)):
    demo_idx = i
    # demo_key = f'data/demo_{demo_idx}/obs/robot_head_cam'
    demo_key = f'data/demo_{demo_idx}/external_camera_obs'
    if demo_key not in file:
        print(f"Demo {demo_idx} not found in file, skipping.")
        continue

    print(f"Replaying {demo_key}...")
    frames = file[demo_key][:]
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_array(frames[frame])
        return [im]
    print(frames.shape)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
    plt.title(f"Demo {demo_idx}")
    # Uncomment to save to file instead of/in addition to showing:
    out_path = f"/datasets/demo_{demo_idx}.gif"
    anim.save(out_path, writer="pillow", fps=1000 / 30)
    plt.show()

# Close the file after all demos are shown
file.close()