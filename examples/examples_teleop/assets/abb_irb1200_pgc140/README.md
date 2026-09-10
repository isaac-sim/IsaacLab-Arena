# ABB IRB1200 + DH PGC-140-50（外扩指尖仿真变体）

完整选型、Docker 运行、验证与限制见 [使用说明](../../../../../../docs/ABB-IRB1200-PGC140-纯物理抓取.md)。

- [vendor/urdf/dh_pgc140_urdf.urdf](vendor/urdf/dh_pgc140_urdf.urdf)：官方原件，未改动。
- [source_manifest.json](source_manifest.json)：固定上游提交、原文件 SHA-256、许可声明与修改说明。
- [pgc140_offset_fingers.urdf](pgc140_offset_fingers.urdf)：派生独立夹爪，两滑块同目标驱动，无硬 mimic。
- [abb_irb1200_pgc140.urdf](abb_irb1200_pgc140.urdf)：复用仓库已有 ABB URDF 的组合模型。
- [gripper/pgc140_offset_fingers.usda](gripper/pgc140_offset_fingers.usda)：独立夹爪 USD。
- [combined/abb_irb1200_pgc140.usda](combined/abb_irb1200_pgc140.usda)：组合机械臂 USD，example39 实际加载此文件。

**不是原装指尖精确 CAD 仿真**：保留厂家壳体和滑动关节；原装内收指尖被 box 滑块与外扩工具替换。原装网格仍在 vendor 中。两块接触垫及其实体支架已写入 USD，不是运行时隐藏代理。

开口 40–90 mm，TCP 为 ABB link_6 局部 `(0.130,0,0)`；总工具质量估计 1.08 kg。本例证明 1 kg 刚体 cube 的接触搬运，不证明薄壁洗衣液瓶、湿表面或真机可靠性。

模型生成物保留在本地工作区；不要把供应商模型或大体积生成资产误提交到上游仓库。