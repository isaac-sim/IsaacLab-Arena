

# cd submodules/Isaac-GR00T
# server需要在5090服务器上运行。拷贝到Submodules/Isaac-GR00T目录下，再运行

export GR00T_DEBUG_LOG=1  # 开启log

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-DROID \
  --embodiment-tag OXE_DROID \
  --device cuda --host 0.0.0.0 --port 5555
