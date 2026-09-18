# Newton SDF 与 Hydroelastic 仿真说明

本目录提供基于 Newton API 的两种网格 SDF 碰撞示例：

- `sdf_basic_collid.py`：基础 SDF 离散点接触。
- `sdf_hydroelastic.py`：基于 SDF 接触面的 Hydroelastic 接触。

两者都会先将 USD 碰撞网格烘焙为稀疏有符号距离场（SDF），并通过
`shape_force_sdf=True` 强制网格碰撞使用该距离场。SDF 烘焙缓存保存到项目内的
`scene/tmp`。Hydroelastic 示例还会对水壶和
纸箱设置 `ShapeFlags.HYDROELASTIC`，并为其设置
`shape_material_kh`（Hydroelastic 刚度）。Hydroelastic 碰撞体必须是水密的封闭网格。

## 运行

```bash
.venv/bin/python examples/sdf/sdf_basic_collid.py
.venv/bin/python examples/sdf/sdf_hydroelastic.py
```

Hydroelastic 示例的 UI 可分别开启接触面、穿透深度、压力和法线的显示；基础 SDF
示例可显示离散接触点与法向力代理值。

## 防止穿模的关键参数

当前两个示例均使用：

```python
FRAME_RATE_HZ = 480
SIM_SUBSTEPS = 4
frame_dt = 1.0 / FRAME_RATE_HZ  # 1/480 s
sim_dt = frame_dt / SIM_SUBSTEPS  # 1/1920 s
```

高速、薄壁或高刚度接触发生穿模时，优先按下列顺序调整：

| 参数 | 作用 | 调整建议与代价 |
| --- | --- | --- |
| `sim_dt` | 单次积分的时间步长 | 减小步长最直接可靠；提高 `FRAME_RATE_HZ` 或 `SIM_SUBSTEPS`，但计算量近似随子步数增加。 |
| `HYDROELASTIC_CONTACT_STIFFNESS` | Hydroelastic 的深度到压力系数 | 增大可减少可见压入量，但会使系统更硬、更难收敛；出现抖动或不稳定时应减小它或同步减小 `sim_dt`。 |
| `CONTACT_STIFFNESS` / `CONTACT_DAMPING` | 常规点接触的刚度与阻尼 | 用于台面等非 Hydroelastic 接触体；增大阻尼可抑制振荡，过大则可能造成数值迟滞。 |
| `*_SHAPE_MARGIN` | 碰撞表面的偏移距离 | 适度增大可为接触保留数值余量；过大将使物体看起来提前接触或悬空。 |
| `*_COLLISION_GAP` | 提前生成接触的距离 | 高速物体可适度增大；过大会改变有效接触距离并增加非必要接触。 |
| `MESH_SDF_MAX_RESOLUTION` | SDF 空间分辨率 | 薄壁、小特征或窄缝必须有足够体素分辨率；提高后会增加烘焙时间、显存和查询成本。 |
| `MESH_SDF_NARROW_BAND_RANGE` | SDF 有效窄带范围 | 应覆盖预期接触与最大单步相对位移附近的距离；范围过窄会降低高速接近时的鲁棒性。 |
| `ccd_iterations` / `sdf_iterations` | CCD 与 SDF 接触迭代上限 | 对快速接近、复杂曲面可提高；计算更慢，且不能替代足够小的时间步。 |
| `iterations` / `ls_iterations` | 接触约束求解迭代 | 增加有助于高刚度、多接触约束收敛；计算更慢。 |

实践原则：先减小 `sim_dt`，再提高 SDF 分辨率与接触迭代；仅在步长与分辨率足够时逐步增大刚度。还应确保 Hydroelastic 网格水密、法线方向和尺度正确，并使
`RIGID_CONTACT_MAX` 足以容纳场景中的接触数量。

## Hydroelastic 可访问的中间结果

Hydroelastic 碰撞管线可通过以下入口读取当前帧接触面：

```python
surface = collision_pipeline.hydroelastic_sdf.get_contact_surface()
face_count = surface.face_contact_count.numpy()[0]
triangles = surface.contact_surface_point.numpy()[: 3 * face_count]
depths = surface.contact_surface_depth.numpy()[:face_count]
```

`sdf_hydroelastic.py` 将这些数据处理为下列开发者可用信息：

| 数据 | 来源或计算方式 | 单位/含义 |
| --- | --- | --- |
| 接触面片数 | `face_contact_count` | marching-cubes 生成的有效三角面数量。 |
| 接触面三角形顶点 | `contact_surface_point` | 世界坐标的接触面几何。 |
| 穿透深度 | `contact_surface_depth`，示例取 `max(-depth, 0)` | m。用于表示局部重叠深度。 |
| 压力 | `HYDROELASTIC_CONTACT_STIFFNESS * penetration` | Pa；本示例采用线性深度—压力模型。 |
| 法线 | 三角形顶点叉积并归一化 | 世界坐标单位方向；方向由接触面三角形绕序决定。 |
| 合力与力矩 | 对各面片压力乘面积、沿法线积分 | 可由上述面片、压力和法线在应用层进一步计算。 |

示例已将穿透深度和压力映射为蓝→青→黄→红的伪彩色，并将法线显示为绿色箭头。每隔
`--log-every` 帧输出的 `Hydroelastic faces` 可用于确认是否真的生成了 Hydroelastic
接触面。

## 与基础 SDF 的区别

基础 SDF 仅产生离散接触点、法线及穿透相关的约束数据，不包含接触面积。因此
`sdf_basic_collid.py` 中的彩色强度为：

```python
normal_force_proxy = CONTACT_STIFFNESS * penetration
```

它仅用于比较接触强弱，不能视为真实压力（Pa）。只有 Hydroelastic 接触面具有面积，
才可以定义局部压力场，并进一步积分得到分布式合力和合力矩。
