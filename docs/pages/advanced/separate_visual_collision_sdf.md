# 视觉模型与实心碰撞模型的分离建模及联合仿真

本文总结一个带独立瓶盖的空心水壶模型从视觉资产到动态 SDF/Hydroelastic
仿真的完整处理方法。核心原则是：**视觉几何负责外观，独立的封闭实心网格负责
碰撞，两者组合在同一个刚体根节点下，但不让视觉网格参与物理接触。**

这种结构适用于瓶、罐、带盖容器和由多个零件组成的刚体。它解决以下常见问题：

- 原始视觉模型由多个 Mesh 组成，不能直接作为单一实体碰撞体；
- 模型内部为空腔或零件之间有缝，无法稳定构建有符号距离场；
- 动态刚体使用三角网格碰撞时被 PhysX 回退为凸包；
- 替换碰撞几何后，原始颜色、材质或纹理消失；
- Newton 同时导入视觉和碰撞 shape，导致重复碰撞或视觉缺失；
- 碰撞源网格、SDF 等值面和接触面共面显示，产生 depth fighting。

## 最终资产和代码

| 文件 | 作用 |
| --- | --- |
| `examples/sdf/blender_make_jug_collision.py` | 在 Blender 中合并瓶身与瓶盖，并用体素重建生成实心水密 OBJ。 |
| `examples/sdf/make_jug_collision_blender.py` | 从 USD 提取部件、调用 Blender，并将结果写回 collision USD。 |
| `scene/fstylejug_a01/fstylejug_a01_solid_collision.usda` | Blender 生成的单一实心水密碰撞代理。 |
| `scene/fstylejug_a01/fstylejug_a01_visual_solid_physx.usda` | 组合原始视觉资产和碰撞代理的最终 Physics USD。 |
| `examples/sdf/sdf_basic_collid.py` | 使用独立碰撞代理进行基础离散 SDF 接触。 |
| `examples/sdf/sdf_hydroelastic.py` | 使用同一代理进行 Hydroelastic 面接触、压力计算和调试显示。 |

资产关系如下：

```text
原始视觉 USD（瓶身、瓶盖、Looks、纹理）
        │
        ├────────────── reference ──────────────┐
        │                                        │
        └─ 提取 Mesh → Blender 体素重建          │
                         │                       │
                         └→ 实心水密 collision USD
                                                 │
                                                 ▼
                              组合 Physics USD：/Jug
                              ├── /Visual
                              └── /Collision
                                         │
                                         ▼
                              Newton ModelBuilder.add_usd
                              ├── Visual shapes：仅渲染
                              └── Collision shape：SDF/Hydroelastic
```

## 为什么需要分离视觉和碰撞模型

视觉模型追求材质、UV、法线和轮廓细节，通常包含多个 Mesh、薄壁结构、内部面和细小
零件。碰撞模型追求封闭、稳定和可控的拓扑复杂度。直接复用视觉网格会产生几个问题：

1. 空心容器的内表面会让 SDF 保留空腔，而当前任务需要把有盖水壶视为一个实心刚体。
2. 瓶身与瓶盖即使看起来接触，也可能是两个独立连通分量，接缝处不构成一个封闭体。
3. 动态三角网格不适合作为普通 PhysX mesh collision。若没有正确的 SDF schema，PhysX
   会报告动态 triangle mesh 不受支持，并回退到 `convexHull`。
4. 用简化碰撞网格替换视觉网格会丢失原始材质绑定和纹理。

因此最终 USD 同时保留两套表示。视觉层不应用 Collision API；碰撞层不承担最终渲染。

## 第一步：检查源 USD

编辑前应使用 `pxr.Usd` 检查以下内容：

- `defaultPrim`、`metersPerUnit` 和 `upAxis`；
- 所有 `UsdGeom.Mesh` 的 prim path；
- 每个 Mesh 的 material binding；
- 是否已经应用 `PhysicsCollisionAPI`；
- 根节点和子节点的变换；
- 瓶身、瓶盖是否是独立几何部件。

本例统一使用米和 Z-up。最终视觉层包含瓶身和瓶盖两个 Mesh，材质绑定仍指向完整
Visual reference 中的 `Looks` prim。不要只引用视觉资产的 Mesh 子树，因为材质目标
经常位于兄弟 `Looks` 节点中；只引用局部子树会使几何存在但颜色和纹理丢失。

## 第二步：在 Blender 中制作实心水密碰撞代理

生成流程由 `make_jug_collision_blender.py` 和
`blender_make_jug_collision.py` 配合完成：

1. 从源 USD 读取瓶身和瓶盖 Mesh，并应用 USD 层级变换。
2. 将各个连通部件临时导出为 OBJ。
3. Blender 导入所有 OBJ，并把 location、rotation 和 scale 烘焙到顶点。
4. 合并所有部件。
5. 使用 voxel remesh 重建外部体积，封闭瓶盖接缝并填充内部空腔。
6. 导出一个不含材质、法线和 UV 的 collision OBJ。
7. 焊接 OBJ 中因 corner、法线或 UV 拆分产生的重复顶点。
8. 检查水密性、修正法线并写入独立 collision USDA。

可直接运行：

```bash
.venv/bin/python examples/sdf/make_jug_collision_blender.py \
    scene/fstylejug_a01/fstylejug_a01_inst_physx.usd \
    scene/fstylejug_a01/fstylejug_a01_solid_collision.usda
```

底层 Blender 脚本也可以独立调用：

```bash
blender -b --python examples/sdf/blender_make_jug_collision.py -- \
    /tmp/jug_body.obj /tmp/jug_cap.obj \
    --output-obj /tmp/jug_collision.obj \
    --voxel-size 0.001
```

`voxel-size` 决定碰撞代理的精度和规模。更小的体素保留更多细节，但会增加三角形数、
SDF 烘焙时间、缓存体积和碰撞计算量。选择体素大小时应以物体最小有效特征为依据，
而不是盲目追求与视觉网格相同的分辨率。

### 水密性验收

仅成功导出文件并不代表碰撞代理合格。至少应确认：

- 结果只有一个预期的连通实体；
- 每条边恰好连接两个三角形；
- 没有边界边、非流形边和退化三角形；
- 法线方向一致；
- 原空腔内部的采样点现在被判断为实体内部；
- 几何坐标、尺度和轴方向与视觉模型一致。

Hydroelastic 示例在构建 SDF 前会再次焊接同位置顶点并统计边使用次数。发现开放边、
非流形边或退化三角形时立即停止，而不是把无效网格继续交给求解器。

## 第三步：组合最终 Physics USD

最终文件 `fstylejug_a01_visual_solid_physx.usda` 使用一个刚体根节点和两个职责清晰的
子节点：

```usda
def Xform "Jug" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    bool physics:kinematicEnabled = 0
    bool physics:rigidBodyEnabled = 1

    def Xform "Visual" (
        prepend references = @./fstylejug_a01_inst_base.usd@</RootNode>
    )
    {
        token purpose = "render"
    }

    def Mesh "Collision" (
        prepend apiSchemas = [
            "PhysicsCollisionAPI",
            "PhysicsMeshCollisionAPI",
            "PhysxSDFMeshCollisionAPI"
        ]
        prepend references = @./fstylejug_a01_solid_collision.usda@</CombinedMesh>
    )
    {
        token physics:approximation = "sdf"
        bool physics:collisionEnabled = 1
        int physxSDFMeshCollision:sdfResolution = 256
        float physxSDFMeshCollision:sdfMargin = 0.001
        int physxSDFMeshCollision:sdfSubgridResolution = 6
        token purpose = "guide"
    }
}
```

这里有四个不可省略的设计点：

- `PhysicsRigidBodyAPI` 只放在公共根节点，使 Visual 和 Collision 随同一个刚体运动；
- `Visual` 引用完整视觉根节点，以保留几何、颜色、材质和纹理依赖；
- 只有 `/Jug/Collision` 应用 Collision API；
- 动态 PhysX SDF 同时需要 `physics:approximation = "sdf"` 和
  `PhysxSDFMeshCollisionAPI`。只写 approximation 可能仍按 triangle mesh 解析。

`purpose = "guide"` 表示碰撞代理不是最终视觉内容。调试器仍可在开启 collision 显示时
把它画出来。

## 第四步：在 Newton 场景中联合导入

`ModelBuilder.add_usd` 会把同一 USD 中的视觉和碰撞 Mesh 都导入为 shape。导入后必须
按 prim path 分流，不能把所有 shape 一起构建成 SDF：

```python
jug_result = builder.add_usd(
    str(JUG_USD_PATH),
    xform=jug_xform,
    schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
)

visual_shapes = []
collision_shapes = []
for prim_path, shape_index in jug_result["path_shape_map"].items():
    if prim_path.endswith("/Collision"):
        collision_shapes.append(shape_index)
    else:
        visual_shapes.append(shape_index)
        builder.shape_flags[shape_index] |= int(newton.ShapeFlags.VISIBLE)
```

Newton 可能不会自动为非 collider Mesh 设置 `VISIBLE`，所以视觉 shape 需要显式打开该
标志。不要为了让视觉模型显示而给它添加 Collision API，否则瓶身、瓶盖和实心代理会
同时参与接触，造成重复碰撞。

随后只对 `collision_shapes` 执行以下操作：

- 设置 `shape_margin` 和 `shape_gap`；
- 设置普通接触的 `ke`、`kd` 和 restitution；
- 调用 `geometry.build_sdf(...)`；
- 检查 `geometry.sdf is not None`；
- 设置 `builder.shape_force_sdf[shape_index] = True`；
- Hydroelastic 模式下再添加 `ShapeFlags.HYDROELASTIC` 和 `shape_material_kh`。

SDF 使用的 scale 必须来自导入后的 `builder.shape_scale[shape_index]`。缓存目录固定到
`scene/tmp`，相同几何和参数可以复用烘焙结果。

## 基础 SDF 与 Hydroelastic 的联合使用方式

两个示例共享同一套 Visual/Collision 分离资产，但接触模型不同：

| 项目 | 基础 SDF | Hydroelastic |
| --- | --- | --- |
| 示例 | `sdf_basic_collid.py` | `sdf_hydroelastic.py` |
| 几何输入 | 独立实心 `/Collision` Mesh | 同一独立实心 `/Collision` Mesh |
| 接触输出 | 离散接触点 | marching-cubes 接触三角面和求解接触 |
| 强度数据 | `ke × penetration` 的法向力代理 | `kh × penetration` 的压力，单位 Pa |
| 适用场景 | 常规刚体接触和快速调试 | 需要接触面积、压力分布和面法向的场景 |
| 水密要求 | SDF 推荐水密 | 必须是封闭水密体 |

两种模式都使用 `CollisionPipeline`、MuJoCo-Warp solver 和相同的仿真子步循环。接触
buffer 容量必须同时传给 model、collision pipeline 和 solver。若关闭 contact reduction
以查看全部原始接触，接触数量可能从几千增至数万，需要同步提高 `rigid_contact_max`、
`njmax` 和 `nconmax`，否则日志会出现 contact buffer overflow。

### 分别设置质量和材料刚度

质量与接触材料属于仿真层参数，不应写死在视觉几何中。Hydroelastic 示例当前使用：

- 水壶质量 `JUG_MASS = 5.0 kg`；
- 水壶普通接触刚度 `JUG_CONTACT_STIFFNESS = 5.0e4`；
- box 普通接触刚度 `BOX_CONTACT_STIFFNESS = 2.5e5`；
- 水壶 Hydroelastic 刚度 `JUG_HYDROELASTIC_CONTACT_STIFFNESS = 2.0e9`；
- box Hydroelastic 刚度 `BOX_HYDROELASTIC_CONTACT_STIFFNESS = 1.0e11`。

修改导入刚体的质量时，必须按质量比例同步缩放完整惯量张量：

```python
mass_scale = target_mass / imported_mass
builder.body_mass[body_index] = target_mass
builder.body_inertia[body_index] *= mass_scale
```

只改质量而不改惯量会使平动和转动响应不一致。水壶和 box 的 `ke`、`kh` 应通过不同
命名参数显式传入，避免共享默认值造成调参误判。

## 调试可视化

### 分离显示碰撞源网格与 SDF 等值面

ViewerGL 的 collision 显示包含两种不同内容：

- 灰色网格：构建 SDF 前的 source collision mesh；
- 黄色网格：烘焙后的 SDF 零等值面。

两者几何接近，若同时显示会产生类似噪点的 depth fighting。这不是视觉 Mesh 残留，
而是两层碰撞调试几何共面。示例增加了两个独立开关，并保留 Viewer 的
`Show Collision` 作为总开关。

独立显隐必须在 `viewer.log_state(...)` 之后执行。`log_state` 每帧可能重新同步对象状态，
若先设置 `hidden`，随后会被统一 collision 显示逻辑覆盖，看起来就像独立开关无效。

### 接触强度伪彩色

基础 SDF 把接触点显示为球形点云，并用蓝、青、绿、黄、红表示相对法向力代理值。
Hydroelastic 把接触三角形的边显示为线框，并可切换接触面、穿透深度、压力和法向。

伪彩色采用正值分位数归一化，而不是简单除以全局最大值。这样少量极端峰值不会把
其余数据压缩成几种标准色。颜色通过逐点或逐线段 RGB 数组传入 ViewerGL。

灰色接触面、深度和压力使用完全相同的三角形边，不能同时绘制，否则会发生共面覆盖。
界面将三者设计为互斥显示模式；法向箭头可以独立叠加。

### 压力沿 SDF 零水平集扩散

压力原始值已经位于 Hydroelastic 提取的 SDF 零水平集接触网格上，因此不需要投影到
原 collision mesh。显示扩散直接在该动态三角网格上进行：

1. 以容差量化三角形顶点，把重复坐标焊接为共享顶点 ID；
2. 通过共享顶点建立隐式面邻接；
3. 把当前面压力累积到顶点并求相邻平均；
4. 将顶点平均回采样到每个三角面；
5. 重复若干轮，再对扩散后的面压力着伪彩色。

每轮扩散可写成：

```text
p_next = (1 - rate) * p_source + rate * neighbor_average(p_current)
```

当前使用 8 轮、`rate = 0.85`。源项保留使高压核心不会在多轮平均后完全消失，压力会
沿零水平集的拓扑邻接向周围衰减传播。该过程只改变调试显示，不修改求解器使用的原始
物理压力。界面分别报告 `Max physical pressure` 和
`Max diffused display pressure`，避免混淆。

## 常见故障与定位

### Visual Mesh 没有显示

确认视觉 prim 已被 `add_usd` 导入，并对非 `/Collision` shape 显式设置
`ShapeFlags.VISIBLE`。不要通过启用 collision 来间接获得可见性。

### 几何可见但材质或纹理丢失

检查组合 USD 是否引用了完整视觉根节点，以及 material binding 的目标 prim 是否仍在
组合 stage 中。只引用 Mesh 子树通常会漏掉兄弟 `Looks` 节点。

### PhysX 回退到 convex hull

若出现“triangle mesh collision cannot be a part of a dynamic body”，检查最终组合 prim，
而不仅是被引用的 collision 文件。有效的 `/Collision` prim 必须同时包含：

- `PhysicsCollisionAPI`；
- `PhysicsMeshCollisionAPI`；
- `PhysxSDFMeshCollisionAPI`；
- `physics:approximation = "sdf"`。

修改后需重新加载 stage，旧的 PhysX cook 缓存不会自动代表新组合结果。

### 灰色和黄色网格同时闪烁

灰色是 source mesh，黄色是 SDF isosurface。关闭其中一个独立开关；必要时关闭总开关
`Show Collision`。这与原始 Visual Mesh 是否显示无关。

### 独立开关只受 Show Collision 控制

确认细分开关逻辑在 `viewer.log_state` 之后运行，并分别找到 `_shape_to_batch` 中的
source mesh instance 和 `_sdf_isomesh_instances` 中的 SDF batch。总开关应参与最终
布尔表达式，但不能替代两个子开关。

### 压力或深度没有渐变色

确认只有一个共面接触显示模式处于开启状态；检查传入 `log_lines` 的颜色数组长度是否
等于线段数；同时检查标量是否存在多个有限正值。极端长尾数据应使用分位数归一化。

### 接触 buffer 溢出

日志中的实际接触计数可能大于分配容量。若任务需要保留全部接触，应关闭 reduction 并
同步增加 collision pipeline 和 solver 容量；若只需要稳定求解而非完整调试点集，则保留
contact reduction 通常更节省显存和计算时间。

## 验收清单

资产侧：

- [ ] 原始视觉 USD 未被覆盖；
- [ ] Visual 包含原始几何、material binding 和纹理依赖；
- [ ] Collision 是单一、封闭、实心、水密网格；
- [ ] Visual 与 Collision 的坐标系、单位和变换一致；
- [ ] 根节点只有一个动态刚体；
- [ ] 只有 `/Collision` 应用 Collision API；
- [ ] `/Collision` 的 PhysX approximation 为 SDF。

仿真侧：

- [ ] `path_shape_map` 被分为 visual shapes 和 collision shapes；
- [ ] visual shapes 只设置 `VISIBLE`，不参与 SDF 和接触；
- [ ] 只有 collision shapes 执行 `build_sdf` 和 `shape_force_sdf`；
- [ ] Hydroelastic 模式下碰撞网格通过水密检查并设置 `HYDROELASTIC`；
- [ ] 水壶和 box 使用显式、独立的 `ke` 与 `kh`；
- [ ] 修改质量时惯量同步缩放；
- [ ] source mesh、SDF isosurface 和接触数据显示可以独立验证；
- [ ] 没有 convex-hull fallback、重复 collider 或 contact buffer overflow。

## 运行示例

基础 SDF 接触：

```bash
.venv/bin/python examples/sdf/sdf_basic_collid.py
```

Hydroelastic 接触与压力可视化：

```bash
.venv/bin/python examples/sdf/sdf_hydroelastic.py
```

建议先只显示 Visual，确认材质与姿态；再依次单独打开灰色 collision source、黄色 SDF
isosurface 和接触数据。逐层检查比同时打开所有调试几何更容易定位模型结构或显示问题。
