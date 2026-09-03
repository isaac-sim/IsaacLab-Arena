# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from lightwheel_sdk.loader import object_loader


"""
examples/example_05_lightwheel_assets.py

示例：列出 Lightwheel 注册表中的资产并从注册表获取（acquire）一个对象文件（例如 USD）。

运行方式：
  1. 激活项目虚拟环境（如使用仓库自带的 .venv）：
     source .venv/bin/activate
  2. 在仓库根目录运行：
     python examples/lightwheel_assets.py

期望结果：
  - 控制台会打印注册表中资产总数以及按 registryType 分组的名称列表；
  - 随后会打印 acquire_by_registry 返回的 `file_path, object_name, metadata`，
    其中 `file_path` 通常为下载或缓存后的本地文件路径（例如 .usd）。

注意：确保 `lightwheel_sdk` 已安装并能访问 Lightwheel registry（网络/权限）。
"""


def list_and_acquire():
    """列出注册表并从 registry 获取一个对象。

    将原有的顶层执行逻辑封装为函数，便于复用和测试。
    """

    # 列出 Lightwheel registry 中所有可用的资产（返回为 dict 列表）
    registry_data = object_loader.list_registry()

    # 按 registryType 分组资产名称以便更清晰地展示
    by_type = defaultdict(list)
    for item in registry_data:
        # 每个 item 形如 {"registryType": ..., "name": ..., ...}
        by_type[item["registryType"]].append(item["name"])

    print(f"=== Lightwheel Registry: {len(registry_data)} total assets ===\n")
    for rtype, names in sorted(by_type.items()):
        print(f"--- {rtype} ({len(names)} assets) ---")
        for name in sorted(names):
            print(f"  {name}")
        print()

    # 从注册表中获取一个对象文件（示例使用 registry 名称 alphabet_soup）
    # 返回值通常是 (local_file_path, object_name, metadata)
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="objects", registry_name=["alphabet_soup"], file_type="USD"
    )

    print("Acquired object:")
    print("  file_path:", file_path)
    print("  object_name:", object_name)
    print("  metadata:", metadata)


if __name__ == "__main__":
    list_and_acquire()
