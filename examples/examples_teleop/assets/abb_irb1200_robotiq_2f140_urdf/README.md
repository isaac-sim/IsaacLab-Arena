# ABB IRB1200 + upstream Robotiq 2F-140

此目录是新下载模型的派生产物，与旧 Collected 2F85、简化两滑块模型和 PGC140 不同。

- 来源：ROS-Industrial [robotiq](https://github.com/ros-industrial-attic/robotiq)，固定提交 `45196f6558fe8ba9d89bc8a105396c68c3e7e892`。
- 原始文件及 [许可证](vendor/LICENSE) 保留；[来源清单](source_manifest.json) 记录 17 个文件的 URL/SHA-256 及派生修改。
- [展开 URDF](robotiq_2f140.urdf) 保留六个 revolute、五条 mimic、原装指腹。
- [夹爪 USD](gripper/robotiq_2f140.usda) 先独立转换。
- [reference 组合](assembly.usda) 引用本地 ABB-only USD 和夹爪 USD，法兰 `Ry(pi/2)` + fixed joint。
- [最终组合 USD](abb_irb1200_robotiq_2f140.usda) 为上述组合的 flatten 结果。保留本地 ABB 源资产目录；不是脱离工作区的独立发布包。
- [抓取脚本](../../example40_newton_abb_irb1200_robotiq_2f140_physical_grasp.py) 只驱动一个主动轴，并显式加载 Newton schema 及设置接触/等式软约束参数。单独用其他脚本打开 USD 不会自动获得同样控制器设置。
- [转换器](../../../../tools/convert_robotiq_2f140_urdf_to_abb_usd.py)、[资产和空载动态验证器](../../../../tools/validate_robotiq_2f140_assets.py)。

本资产保留上游可视化模型惯量，补两个 10 g pad 后总质量约 0.38385 kg，**不是标称产品质量校准值**。内开口实测从 128.5 mm 到约零；不是强行缩放到标称 140 mm。力矩 5 N·m、摩擦 0.8 和接触刚度为仿真参数，不是真机额定值。

详见 [中文原理、运行与验证说明](../../../../../../docs/ABB-IRB1200-Robotiq2F140-原生联动纯物理抓取.md)。模型/网格仅作本地资产保存，不提交大模型文件。