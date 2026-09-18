# IsaacLab-Arena 代码库深度分析报告

> 分析对象：`isaaclab_arena/` 下的 `assets/`、`scene/`、`relations/`、`affordances/`、`embodiments/`、`terms/`、`teleop/`、`utils/`，以及环境组合链路 `environments/`、`environment_spec/`、`tasks/`。
> 基线：Isaac Sim 6.0 + Isaac Lab 3.0 Beta，仓库 alpha（v0.2.x）。

---

## 0. 总体架构

IsaacLab-Arena 是一个**可组合环境创建 + 策略评估库**，核心思想：把环境拆成三种独立原语——**Scene（场景/资产）、Embodiment（机器人具身）、Task（任务）**，用统一的 configclass 合并机制把它们"编译"成一个 Isaac Lab ManagerBased 环境。其上有两层：

1. **声明层**：`environment_spec/`（YAML 环境图 `ArenaEnvGraphSpec`）或编程式 `IsaacLabArenaEnvironment`；
2. **编译层**：`environments/arena_env_builder.py` 的 `ArenaEnvBuilder`，负责空间关系求解、变体注入、把 Scene/Embodiment/Task 各自产出的 `get_*_cfg()` 用 `combine_configclass_instances` 合并成 `IsaacLabArenaManagerBasedRLEnvCfg`，注册进 gym 并 `make()`。

三层调用链：`ArenaEnvGraphSpec(YAML) → IsaacLabArenaEnvironment → ArenaEnvBuilder.compose_manager_cfg() → gym.register + IsaacLabArenaManagerBasedRLEnv`。

---

## 1. `assets/` 模块（资产体系）

### 1.1 资产类层次（核心文件）

**`asset.py` — `Asset` 基类**
- 用途：一切资产（物体/集合/背景/光源）的最顶层基类，承载**命名、标签（tags）、变异（variations）**。
- API：`__init__(name, tags)`、`add_variation()` / `get_variation()` / `get_variations()`（挂载 `VariationBase`，如 RIGID 物体自动挂 `ObjectMassVariation`）、`get_scene_key()`（默认返回 name）。采用协作式多重继承（`super().__init__(**kwargs)`），支持 `Object` + 行为 mixin 的组合。

**`object_type.py` — `ObjectType` 枚举**
- 用途：`BASE` / `RIGID` / `ARTICULATION` 三态类型。刻意零依赖（不 import pxr），避免 pytest 收集期在 `SimulationApp()` 启动前拉起 USD 导致段错误；被 `object_base.py` 重导出作为单一事实来源。

**`relations/placement_asset.py` — `PlaceableAsset`（assets 与 relations 的桥）**
- 用途：可被空间关系求解器约束根位姿的资产抽象，`ObjectBase` 与 `EmbodimentBase` 的共同基类。
- API：`add_relation()` / `get_relations()` / `get_spatial_relations()`（过滤标记类，只留 `Relation`/`UnaryRelation`）、`is_anchor`、`requires_reachability`、`set_initial_pose(pose, create_reset_event)`（支持 `Pose`/`PoseRange`/`PosePerEnv`）、`_build_reset_event()`、`layout_pose_to_scene_writes()`、抽象 `get_bounding_box()`、`get_world_bounding_box()`、`get_collision_mesh()`、`collision_mode`。

**`object_base.py` — `ObjectBase(PlaceableAsset, ABC)`**
- 用途：`Object` 与 `ObjectReference` 的共同父类，负责 prim 路径、类型分发、初始位姿/速度、reset 事件、场景读写。
- API：默认 `prim_path = "{ENV_REGEX_NS}/" + name`；`_set_initial_pose()` / `set_initial_velocity()`（写入 `object_cfg.init_state`）；`_build_reset_event()` 按位姿类型分派——`Pose`→`terms.events.set_object_pose`、`PosePerEnv`→`set_object_pose_per_env`、`PoseRange`→`isaaclab_tasks` 的 `randomize_object_pose`；`get_object_cfg()` 返回 `(name, RigidObjectCfg|ArticulationCfg|AssetBaseCfg)`（Scene 消费资产的核心入口）；`get_event_cfg()` 返回 `(name, EventTermCfg|None)`；`_init_object_cfg()` 按类型分派到抽象方法 `_generate_rigid_cfg/_generate_articulation_cfg/_generate_base_cfg`；`get_object_pose()` / `set_object_pose()`（经 `env.scene[name]` 读写 sim）；`get_contact_sensor_cfg()`。

**`object.py` — `Object(ObjectBase)`**
- 用途：真正的可生成物体，绑定 USD 文件或自定义 spawner，产出 Isaac Lab `*Cfg`。
- API：`usd_path` 与 `spawner_cfg` 二选一；`object_type` 缺省时 `detect_object_type(usd_path)` 自动探测；`scale`、`relations`、`spawn_cfg_addon`/`asset_cfg_addon`；`get_bounding_box()` / `get_corners()`（从 USD 算局部 AABB，惰性缓存）；`disable_reset_pose()`/`enable_reset_pose()`；`get_contact_sensor_cfg()` 覆写——用 `find_shallowest_rigid_body` 把接触传感器挂到 USD 中最浅刚体；`_get_spawn_cfg()` 产出 `UsdFileCfg(usd_path, scale, ...)`。

**`object_reference.py` — `ObjectReference(ObjectBase)` + `OpenableObjectReference`**
- 用途：**不生成新物体**，引用父资产 USD 内已存在的 prim（如门把手、微波炉托盘）；放置求解器把它当固定点。
- API：`parent_asset`；`get_initial_pose()`（`T_W_O = T_W_P * T_P_O` 组合）；`add_relation()` 只接受 `IsAnchor`；`get_bounding_box()` 从父 USD 的 prim 计算；静态方法 `isaaclab_prim_path_to_original_prim_path()`（把 `{ENV_REGEX_NS}/...` 路径还原为 USD 内路径）；`OpenableObjectReference(ObjectReference, Openable)` 混入 `Openable` 行为。

**`object_set.py` — `RigidObjectSet(Object)`**
- 用途：一组刚性物体，**每个环境各选一个成员**生成（Isaac Lab `MultiUsdFileCfg`）。
- API：`__init__(name, objects, random_choice, ...)`；USD 预处理——`_modify_assets()` → `rescale_rename_rigid_body_and_save_to_cache()`（统一 scale/刚体命名后缓存到磁盘）；`assign_variants(num_envs, variant_seed)`（固定每 env 成员索引，可随机/循环）；`get_bounding_box_per_env(num_envs)`（逐 env 真实变体 bbox，供异构放置）；`_generate_rigid_cfg()` 用 `MultiUsdFileCfg(usd_path=object_usd_paths, random_choice=False)`。

**`background.py` — `Background(Object)`**
- 用途：环境背景场景资产，是 `Object` 的特化——无物理（`ObjectType.BASE`），复用整套 object 体系。
- API：`__init__(name, usd_path, object_min_z, ...)`（`object_min_z` 是物体掉落 reset 用的全局 Z 高度）；`get_viewer_cfg()` 返回背景专属相机取景。

**`object_library.py` — 预置资产库（1964 行）**
- 用途：声明约 150 个开箱即用资产类，每个类以**类属性**描述资产并经 `@register_asset` 注册进 `AssetRegistry`。
- 结构：`LibraryObject(Object)` 基类（name/tags/usd_path/object_type/scale/spawn_cfg_addon/asset_cfg_addon 全部类属性，实例化时 `instance_name` 缺省取 `self.name`）。分类：
  - 基础 YCB：`CrackerBox`、`MustardBottle`、`SugarBox`、`TomatoSoupCan`、`PowerDrill`；
  - **行为混入（多重继承）**：`Microwave(LibraryObject, Openable)`、`CoffeeMachine(..., Pressable)`、`StandMixer(..., Turnable)`、`Mug(..., Placeable)`；
  - 装配件：`Peg`/`Hole`/`SmallGear`/`LargeGear`（ARTICULATION，配 `RIGID_BODY_PROPS_HIGH/MEDIUM_PRECISION`）；
  - 光源：`LightBase(LibraryObject, ABC)` → `DomeLight`（支持 `add_hdr()` + 光变异）、`DirectionalLight`（`set_orientation()`）；
  - 过程式：`GroundPlane`、`Sphere`、`ProceduralTable`、`ProceduralCube`（直接继承 `Object`，用 `CuboidCfg` 生成，不依赖 USD）；
  - Lightwheel 懒加载：`Broccoli`、`SweetPotato`、`Jug`、`BeerBottle`（`LightwheelLazyPath`）；
  - Robolab SRL 系列：`*BasicRobolab`（积木）、`*FruitsVeggiesRobolab`（果蔬）、`*HandalRobolab`（厨具）、`*HopeRobolab`（食品）、`*Hot3DRobolab`、`*ObjaverseRobolab` 等，均带 tag（如 `["object","graspable","food","robolab"]`）供按 tag 检索。

**`asset_cache.py`** — `get_arena_asset_cache_dir()` 返回 `~/.cache/isaaclab_arena/assets`，供 USD 落盘缓存（object set、robot-on-stand）。

**继承关系总览**
```
Asset (asset.py)
└── PlaceableAsset (relations/placement_asset.py)
    ├── ObjectBase (object_base.py, ABC)
    │   ├── Object (object.py)
    │   │   ├── LibraryObject (object_library.py) ── ~150 个 @register_asset 具体类
    │   │   │     （ProceduralTable/ProceduralCube 直接继承 Object）
    │   │   └── RigidObjectSet (object_set.py)
    │   └── ObjectReference (object_reference.py)
    │       └── OpenableObjectReference (+ Openable)
    └── EmbodimentBase (embodiments/embodiment_base.py) ── 各机器人
Background(Object, ObjectType.BASE)
```

### 1.2 注册与发现（registries / register）

**`registries.py` — 注册表中枢**
- 用途：定义全部单例注册表，注册采用**懒加载**——库模块未导入前注册表为空，首次访问由 `ensure_assets_registered()` 统一导入各 library 模块触发装饰器注册。
- API：`Registry(metaclass=SingletonMeta)`（`register`/`is_registered`/`get_component_by_name`/`get_all_keys`）；子类：`AssetRegistry`（含 `get_assets_by_tag`/`get_assets_with_all_tags`/`get_random_asset_by_tag`）、`DeviceRegistry`、`RetargeterRegistry`（键 `"device__embodiment"`）、`PolicyRegistry`、`HDRImageRegistry`、`EnvironmentRegistry`、`ObjectRelationLibraryRegistry`（关系类）、`TaskRegistry`（任务类）。
- `ensure_assets_registered()`：导入 `background_library`、`device_library`、`hdr_image_library`、`object_library`、`retargeter_library`、`embodiments`、`policy`、`relations.relations`、`tasks.task_library`；用 `_assets_registered` 与 `_registration_in_progress` 防重入/防循环（避免 pytest 收集期提前 import pxr 段错误）。

**`register.py` — 声明式注册装饰器**
- API：`@register_asset`（键 `cls.name`）、`@register_device`、`@register_retargeter`（复合键）、`@register_policy`（从 `PolicyBase[ConcreteCfg]` 泛型基类推断 cfg 类型）、`@register_hdr`、`@register_environment`（从 `ArenaEnvironmentFactory[ConcreteCfg]` 推断）、`@register_object_relation`、`@register_task`（键 `cls.__name__`，便于 YAML 查找）、`agent_ready`。重复注册只打 WARNING；一律 `ensure_loaded=False` 避免 import 中途全量加载重入。

### 1.3 资产库与资源源

- **`background_library.py`**：`LibraryBackground(Background)` + `@register_asset`（背景与物体**共用同一 AssetRegistry**）：`KitchenBackground`、`PackingTableBackground`、`Table`、`MapleTableRobolab`、`LightwheelKitchenBackground`（`acquire_lightwheel_asset` 懒下载 Robocasa 厨房）等。
- **`hdr_image.py` / `hdr_image_library.py`**：`HDRImage(Asset)`（`texture_file`/`texture_format`）；`LibraryHDR` + `@register_hdr` 注册 11 个 HDR（`HomeOfficeHDRRobolab`、`EmptyWarehouseHDRRobolab` 等），供 DomeLight 使用。
- **`device_library.py`**：`TeleopDeviceBase(ABC)`；`OpenXRCfg`（装配 `IsaacTeleopCfg`）、`KeyboardCfg`（`Se3KeyboardCfg`）、`SpaceMouseCfg`（`Se3SpaceMouseCfg`），均 `@register_device`。
- **`retargeter_library.py`**：`RetargetterBase(ABC)`（`device`/`embodiment` 类属性）；`FrankaIKIsaacTeleopRetargeter`（openxr__franka_ik → `teleop.single_arm_openxr_pipeline`）、`GR1T2PinkIsaacTeleopRetargeter`、`G1WbcPinkIsaacTeleopRetargeter` 等；键盘/鼠标重定向器返回 None（无需管线）。
- **`nucleus.py`**：`ARENA_NUCLEUS_DIR`——把 `ISAACLAB_NUCLEUS_DIR` 的 `omniverse-content-production` 替换为 `omniverse-content-staging`（暂存桶，发布前切回生产）。
- **`lightwheel_lazy.py` / `lightwheel_utils.py`**：`LightwheelLazyPath` 描述符实现类属性级懒加载（首次访问经 lightwheel SDK `acquire_by_registry` 下载 USD 并缓存）；`acquire_lightwheel_asset()` 提供统一超时（改写 `client.base_timeout`）+ 重试语义。
- **`object_utils.py`**：`detect_object_type()`（BFS 在**最浅深度**找 `RigidBodyAPI`/`ArticulationRootAPI`，同层多个报错，无则 BASE）；`RIGID_BODY_PROPS_HIGH/MEDIUM_PRECISION`；`EMPTY_ARTICULATION_INIT_STATE_CFG`（空关节 dict，避免命中 `{".*": 0.0}` 默认模式）。

### 1.4 assets 与其他模块的交互

- **Scene**（scene.py）：`asset.get_object_cfg()` → 场景 cfg 字段；`asset.get_event_cfg()` → reset 事件；`get_objects_with_relations()` 筛出带关系的 Object/ObjectReference 交给 relation 求解器。
- **relations**：`PlaceableAsset` 是求解器与资产之间的契约；`ObjectReference`/`Background` 被 `passive_collision_objects`/`background_collision_object` 用作碰撞体。
- **embodiments**：机器人也经 `@register_asset` 注册进 `AssetRegistry`（embodiment 也是资产）。
- **variations**：`Asset.variations` 挂载 `VariationBase`（质量/光强/HDR/相机内外参变异），由 `ArenaEnvBuilder` 收集成 Hydra 可配置项。

---

## 2. `relations/` 模块（自然语言空间关系求解）

> ⚠️ 重要澄清：**本模块没有自然语言解析器**。"on top of / next to / inside" 等短语是关系类的语义（docstring 级对应），关系是**程序化构造**的（`mug.add_relation(On(table))`），每个关系类用 `@register_object_relation` 以 `cls.name`（`"on"`、`"next_to"`…）注册进 `ObjectRelationLibraryRegistry`。

### 2.1 关系声明层 — `relations.py`

- `Side(str, Enum)`：`POSITIVE_X/NEGATIVE_X/POSITIVE_Y/NEGATIVE_Y`。
- 基类：`RelationBase`（唯一方法 `validate_placement_configuration(subject, objects)` 做放置前校验）。
- 一元：`UnaryRelation`（`is_unary()==True`）→ `AtPosition(x,y,z 可选)`、`PositionLimits(各轴 x_min/x_max 可选)`。
- 二元：`Relation(parent, relation_loss_weight=1.0)` → `On`（`clearance_m=0.01`、`edge_margin_m=0.05` 防落边）、`NextTo`（`distance_m`、`side`、`cross_position_ratio`、`tolerance_m`）、`NotNextTo`（禁止某侧半平面）。
- 标记/朝向类（直接继承 `RelationBase`）：`FaceTo`（+X 朝向目标，不约束位置）、`IsAnchor`（固定参照）、`RequiresReachability`（机器人可达性标记，求解忽略、校验拒绝）、`RandomAroundSolution`（解算位置转 `PoseRange` 做重置随机化）、`RotateAroundSolution`（解算位置叠加显式旋转）。
- 工具：`get_anchor_objects(objects)`、`get_relation(obj, type)`。

### 2.2 损失与求解层

**`relation_loss_strategies.py`**（621 行）
- 策略层次：`UnaryRelationLossStrategy` → `AtPositionLossStrategy`（逐轴钉点）、`PositionLimitsLossStrategy`；`RelationLossStrategy` → `NextToLossStrategy`（半平面+带+距离）、`OnLossStrategy`（XY 带 + Z 点）、`NotNextToLossStrategy`（`min(remaining_side, remaining_cross)`）；`NoCollisionLossStrategy`（三轴重叠体积 × slope，求解器内置 slope=10000）。
- 同时暴露 `next_to_violations()` / `not_next_to_violations()` 供放置校验器复用——**保证"求解最小化什么、校验就检查什么"**。

**`loss_primitives.py`** — 4 个 ReLU 风格可微原语：`single_boundary_linear_loss`、`linear_band_loss`、`single_point_linear_loss`、`interval_overlap_axis_loss`。

**`relation_solver_params.py`** — `RelationSolverParams`：`max_iters=600`、`lr=0.01`、`convergence_threshold=1e-4`、`collision_mode=BBOX`（默认；可选 MESH）、`strategies` 默认注册表（On slope=100 / NextTo slope=10 / NotNextTo slope=10 / AtPosition / PositionLimits）。

**`relation_solver_state.py`** — `RelationSolverState(objects, initial_positions, env_bboxes, collision_objects)`：锚点冻结（跨环境一致性断言）、可优化位置张量 `(batch, N, 3)` 挂 `requires_grad`；锚点/障碍世界包围盒一次性缓存。

**`relation_solver.py`** — `RelationSolver`
- `solve(objects, initial_positions, env_bboxes, orientations, collision_objects)`：策略模式，按 `type(relation)` 查策略；**纯 PyTorch + Adam** 循环（zero_grad → 总损失 → 有 grad_fn 才 backward/step → 收敛判断 `1e-4`）最小化"关系损失 + 自动加入的 no-overlap 损失"；返回每环境 `dict[PlaceableAsset, (x,y,z)]`。
- 诊断：`last_loss_history`、`last_position_history`、`debug_losses()`。

### 2.3 网格碰撞（MESH 模式，可选）

- `collision_mode.py`：`CollisionMode.BBOX`（快速保守）/ `MESH`（sphere-to-SDF 精确）；`get_object_collision_mode()`。
- `no_overlap_aabb.py`：`compute_no_overlap_loss_aabb()`（batched 可导；`NoOverlapPair` 有向对使梯度只流向 subject；跳过 On 对）。
- `warp_mesh_manager.py`：`WarpMeshAndSphereCache`（从 USD 提取碰撞网格 + `wp.Mesh` BVH 缓存 + 贪心球分解 `greedy_sphere_decomposition`；非水密网格默认换凸包保证 SDF 符号）。
- `warp_sdf_kernels.py`：`@wp.kernel` 的 `_sdf_query_kernel`/`_multi_mesh_sdf_kernel`（`wp.mesh_query_point_sign_normal` 求带符号距离）；`multi_mesh_sdf()`/`mesh_sdf()` 经 `torch.autograd.Function` 桥回 PyTorch 梯度图；SDF sentinel（1e5/1e6）由 `clamp_sdf_sentinel` 处理为常数零梯度惩罚。
- `mesh_pair_cache.py`：`MeshPairEntry`/`MeshPairCache`（预计算的有向 sphere→mesh 对，单次 kernel 启动）。
- `no_overlap_mesh.py`：`compute_no_overlap_loss_mesh()`（AABB+max_radius 预过滤 → `multi_mesh_sdf` → `relu(radius+clearance-sdf)`）、`prepare_mesh_collision_cache()`。

### 2.4 放置与校验层

**`object_placer_params.py`** — `ObjectPlacerParams`：`solver_params`、`max_placement_attempts=10`、`placement_seed`、`resolve_on_reset`、`min_unique_layouts_per_env=5`、`allow_best_loss_fallbacks`、`enabled_checks`/`required_checks`、`reachability_config`（`ReachabilityConfig`，转发给 cuRobo 扩展构建 `ik_reachable` 检查）。

**`object_placer.py`**（751 行）— `ObjectPlacer`
- 流程：`_prepare_placement`（关系/锚点/初始位姿断言）→ `assign_variants_for_envs` + `build_per_env_bounding_boxes` → 逐候选生成种子化初始位姿/朝向（On 引导采样 + 随机 yaw）→ `_rotate_candidate_bboxes`（yaw 烘进保守 AABB）→ `RelationSolver.solve()` 一次批量求解 → 补 `FaceTo` yaw 并重建包围盒 → `_validate_candidates`（廉价全量 + 昂贵门控）→ `_rank_candidates`（按"必需失败数、可选失败数、loss"升序）→ 取每 env 最优 → `_apply_poses` 写回 `set_initial_pose`（单 env `Pose`/多 env `PosePerEnv`，合成 marker 旋转，支持 `RandomAroundSolution`）。
- 关键点：**没有显式重试循环**——`max_placement_attempts` 个候选一次性批量求解靠排名淘汰，全无效时取最低 loss 作 fallback；产物 `PlacementCandidate` / `PlacementResult`。

**`placement_validators.py`**（725 行）— 校验器链
- `PlacementValidator`(ABC)：`check`（ClassVar 检查名）、`run_after_inexpensive_checks`（昂贵开关）、`is_available(params)`、抽象 `validate_batch(...)`。
- `build_validators(params)` 按注册顺序实例化并过滤 `enabled_checks`。
- 内置子类（均 `@register_validator` → `PlacementValidatorRegistry`）：`OnRelationValidator`（悬浮/跌落：Z 带 + XY footprint）、`NextToValidator`/`NotNextToValidator`（与损失策略共享违规函数）、`FaceToValidator`、`NoOverlapValidator`（AABB 快速短路 + MESH 模式降级 sphere-to-SDF 穿透测试；跳过 On 对与锚点-锚点对）。`IK_REACHABLE` 由外部 cuRobo 扩展注册。

**`placement_validation.py`** — `PlacementCheck` 枚举（`NO_OVERLAP`、`ON_RELATION`、`NEXT_TO`、`NOT_NEXT_TO`、`FACE_TO`、`PHYSICS_SETTLED`、`IK_REACHABLE`）；`PlacementValidationResults`（`do_all_required_validation_checks_pass`、`get_number_of_required_and_optional_failures`、`add_validation_check`）。

**`placement_events.py`** — 运行期放置事件
- 关键点：**不是"放置前/后钩子"，而是 Isaac Lab reset 事件与 sim 写入层**。`solve_and_place_objects(env, env_ids, assets, placement_pool)` 注册为 `EventTermCfg(mode="reset")`（事件名 `PLACEMENT_RESET_EVENT_NAME = "placement_reset"`）：重置时按绝对 env id 从池消费布局，`write_layout_to_sim()` 写 root pose + 零速度（含 `env_origins` 偏移）；`get_placement_pool()` 供离线校验取回池；`get_pose_from_layout()` 合成 marker 旋转 + 布局 yaw。

**`pooled_object_placer.py`** — `PooledObjectPlacer`
- 维护每环境一个布局队列（`EnvLayoutPool`），池耗尽自动重新求解补池；`sample_without_replacement(count)`（消费式）/ `sample_for_envs(env_ids)`（reset 用）/ `sample_with_replacement(count)`（可复现随机抽）；`had_fallbacks` 标志；严格有效布局优先、最后一批允许 best-loss 回退。

**`placement_pool_validation.py`** — 离线物理沉降校验
- `validate_pool_layouts(env, placement_pool, settle_params)`：把池中每布局写入 sim → `utils.physics_settle.step_physics` 步进 `num_steps × decimation` 物理子步 → 读回速度判断沉降 → 把 `PHYSICS_SETTLED` stamp 到布局校验结果。

**`physics_settle_params.py`** — `PhysicsSettleParams`：`num_steps=5`、`lin_vel_thresh=0.1`、`ang_vel_thresh=0.1`。

### 2.5 碰撞对象模型

- `collision_object.py`：`CollisionObject` 是 **Protocol（结构类型）**——求解器/校验器只依赖该接口（name/collision_mode/get_initial_pose/get_bounding_box/get_world_bounding_box/get_collision_mesh）。
- `passive_collision_objects.py`：`get_passive_collision_objects(assets, include_background)`——扫描**无关系、位姿固定（Pose）**的资产作为被动障碍（放置时避让，但从不被优化、不参与关系约束）。
- `background_collision_object.py`：`FixedCollisionObject`（身份位姿 + 已烘世界坐标网格的整场景障碍）；`make_fixed_collision_objects(objects)` 用 `WarpMeshAndSphereCache` 把多对象网格合并成单个 MESH 障碍。

### 2.6 数据流总结（自然语言 → 关系 → 损失 → 求解 → 放置 → 校验 → 沉降）

```
声明:  mug.add_relation(On(table)); table.add_relation(IsAnchor())
        （"on top of" 等短语是类语义；无运行时 NL 解析）
   ↓
映射:  RelationSolverParams.strategies[type(relation)] → 损失策略
        （OnLossStrategy/NextToLossStrategy/... 基于 4 个 loss_primitives 原语）
   ↓
状态:  RelationSolverState 打包 (batch, N, 3) 可微位置张量，锚点冻结
   ↓
求解:  RelationSolver + Adam（关系损失 + no-overlap 损失；默认 AABB，可选 warp 网格 SDF）
   ↓
放置:  ObjectPlacer 批量候选 → 排名取优（校验器链：On/NextTo/NotNextTo/FaceTo/NoOverlap/IK）
        → _apply_poses 写回 set_initial_pose（Pose / PosePerEnv / PoseRange）
   ↓
池化:  PooledObjectPlacer 按绝对 env id 存布局池，自动补池
   ↓
运行:  placement_events.solve_and_place_objects（reset 事件）按 env 消费布局写 sim
   ↓
沉降:  placement_pool_validation 离线物理步进校验 PHYSICS_SETTLED
```

---

## 3. `scene/` 模块 — `Scene` 类

**`scene/scene.py`**（217 行）
- 用途：**扁平资产注册表 + cfg 聚合器**——组织 embodiment 之外的所有资产（objects、backgrounds、object references），并产出 Isaac Lab 可消费的配置片段。
- API：
  - `__init__(assets=None)`：`self.assets: dict[str, Asset | RigidObjectSet]`；可选的 `observation_cfg/events_cfg/termination_cfg/rewards_cfg/curriculum_cfg/commands_cfg` 槽位。
  - `add_asset(asset)` / `add_assets(list)`：按 name 存字典；`add_assets` 断言 `ObjectReference` 的 `parent_asset` 也在场，并**把 `ObjectReference` 排序到父资产之后**加入。
  - `get_scene_cfg()`：对每个资产调 `asset.get_object_cfg()` 得到 `(name, IsaacLabCfg)`，用 `make_configclass("SceneCfg", fields)` 动态生成 configclass——**字段=资产名、值=Isaac Lab cfg**。
  - `get_events_cfg()`：聚合每个资产的 `get_event_cfg()`（reset 位姿事件）。
  - `get_observation_cfg()` / `get_termination_cfg()` / `get_rewards_cfg()` / `get_curriculum_cfg()` / `get_commands_cfg()`：返回 Scene 级槽位。
  - `get_asset_variations()`：`{asset_name: [VariationBase,...]}`。
  - `get_objects_with_relations()`：筛出带空间关系（或 anchor）的 `Object`/`ObjectReference`，交给放置求解器。
  - `export_to_usd(output_path, root_prim_path="/World")`：把场景导出为扁平化 USD（每个资产一个 prim + reference + 初始位姿 + 接触报告 API）。
- 与其他模块交互：被 `ArenaEnvBuilder` 消费（`scene.get_scene_cfg()` 与 `InteractiveSceneCfg`、embodiment、task 的 cfg 合并成最终 `env_cfg.scene`）；embodiment **不在** Scene 里——它单独作为 `IsaacLabArenaEnvironment.embodiment` 存在，由 builder 合并。

---

## 4. `affordances/` 模块（可操作属性接口）

**`affordance_base.py` — `AffordanceBase(ABC)`**
- 用途：所有可操作行为的基类；**必须与 `Asset` 多重继承**（构造时运行时断言 `isinstance(self, Asset)`），从而访问资产的 name 等属性。
- 用法：`class Microwave(LibraryObject, Openable)`。

**`openable.py` — `Openable`**
- 用途：可开合对象接口（门、抽屉、微波炉）。
- API：`openable_joint_name`、`openable_threshold`；`get_openness(env)`（归一化关节位置）、`is_open()`/`is_closed()`（阈值判断）、`rotate_revolute_joint(env, env_ids, percentage)`（`open()`/`close()` 是其别名）；内部把 `joint_names=[openable_joint_name]` 注入 `SceneEntityCfg`。

**`placeable.py` — `Placeable`**
- 用途：可直立放置对象接口（杯、瓶），判断/改写刚体姿态使直立轴对齐世界 +Z。
- API：`upright_axis_name`、`orientation_threshold`；`is_placed_upright(env)`（世界系直立轴与 +Z 夹角 < 阈值）；`place_upright(env, env_ids, upright_percentage)`（`set_normalized_object_pose` 直接在 `(upright轴, +Z)` 平面内旋转根姿态）；辅助函数 `get_object_axis_in_world_frame()`、`_compute_target_quaternions()`。

**`pressable.py` — `Pressable`**
- 用途：可按压对象接口（按钮）。
- API：`pressable_joint_name`、`pressedness_threshold`；`is_pressed(env)`、`press(env, env_ids, percentage)`、`unpress(...)`。

**`turnable.py` — `Turnable`**
- 用途：带离散档位的旋钮/拨盘接口。
- API：`turnable_joint_name`、`min/max_level_angle_deg`、`num_levels`；`get_turning_level(env)`（返回 [-1, num_levels-1]，-1 为死区）；`turn_to_level(env, env_ids, target_level)`；`is_at_level(...)`。

- 与其他模块交互：被 `object_library.py`（`Microwave+Openable` 等）、`object_reference.py`（`OpenableObjectReference`）混入使用；实现依赖 `utils/joint_utils.py`（归一化/反归一化关节读写）。

---

## 5. `embodiments/` 模块（机器人具身）

### 5.1 核心抽象 — `embodiment_base.py`（246 行）

- 用途：所有具身的统一抽象——封装资产加载（USD 摆放、包围盒、碰撞网格）、关节初始化、动作/观察/事件/奖励/课程/命令配置、相机、mimic 数据环境、XR 遥操作配置。
- **继承**：`EmbodimentBase(PlaceableAsset)`——天然具备可摆放资产能力。
- 类属性：`name`（构造断言非空）、`tags=["embodiment"]`、`default_arm_mode`。
- 子类填充的配置属性（基类 None）：`scene_config`（内含 `robot: ArticulationCfg`）、`camera_config`、`action_config`、`observation_config`、`event_config`、`reward_config`、`curriculum_config`、`command_config`、`mimic_env`、`xr`、`termination_cfg`。
- 关键方法：
  - `get_scene_cfg()`：若有初始位姿则写回 `robot.init_state.pos/rot`；`enable_cameras` 时合并 `get_camera_cfg()`。
  - `get_observation_cfg()`：开相机时合并 `make_camera_observation_cfg(camera_config)`。
  - `get_events_cfg()`：把 `_pose_event_cfg`（`_build_reset_event()` 产物，`reset_placement_asset_pose`/`reset_placement_asset_pose_per_env`）合并到事件表**最后**，保证位姿复位晚于关节/根复位。
  - `get_bounding_box()`/`get_collision_mesh()`：从 `scene_config.robot.spawn.usd_path` 惰性计算（供关系放置）。
  - `get_scene_key()` 固定返回 `"robot"`；`get_ee_frame_name(arm_mode)`；`get_teleop_target_frame_prim_path()`（XR 重基准）；`get_camera_cfg()`（断言继承 `ArenaCameraCfg`）；`add_camera_variations()`（注册相机内外参变异）；`set_joint_initial_pos()`。
- 设计特点：**无 `@abstractmethod`**——"约定式抽象"（子类填配置 + name 断言），与 Scene 的绑定契约是 scene key `"robot"`。

### 5.2 各机器人实现对比

| 维度 | Franka | Droid | G1 | GR1T2 | Galbot | Agibot | Kuka Allegro | NoEmbodiment |
|---|---|---|---|---|---|---|---|---|
| 形态 | 单臂+二指 | 单臂+Robotiq 2F-85 | 人形 29DOF | 人形 | 双臂(仅左臂) | 双臂 | 臂+Allegro 手 | — |
| 默认 ArmMode | SINGLE_ARM | SINGLE_ARM | DUAL_ARM | RIGHT | LEFT | LEFT | SINGLE_ARM | — |
| 动作空间 | IK 增量 7D / 关节 8D | IK/相对/绝对关节 8D | 解耦 WBC+关节或 PINK IK | 关节 36D / PINK IK | RMPFlow+夹爪 | RMPFlow+夹爪 | 相对关节位置 | 空 0D |
| EEF | `panda_hand` | Robotiq `base_link` | 双腕 `*_wrist_yaw_link` | `left/right_hand_roll_link` | `left_gripper_tcp_link` | `gripper_center` | `palm_link` | — |
| 台座 | ✅ 0.8755m | ✅ 1.35m 可调 | ❌ | ❌ | ❌ | ❌ | ❌ | — |
| 相机 | 手腕 84² | 外部×2+手腕 720p | 头 640×480 | 头 512² | ❌ | ❌ | ❌(断言禁止) | — |
| MimicEnv | ✅ 7D | ❌(数据模式) | ✅ 17D 含 body | ✅ 36D | ❌ | ✅(继承 Franka) | ❌ | — |
| XR | 基准 prim | — | 骨盆锚定 | 骨盆锚定 | — | — | — | — |

- **Franka**（`franka/franka.py`）：`FrankaIKEmbodiment`(`franka_ik`，`DifferentialInverseKinematicsActionCfg`+`BinaryJointPositionActionCfg`)、`FrankaJointPosEmbodiment`(`franka_joint_pos`)；`FrankaSceneCfg`（robot ArticulationCfg + `ee_frame: FrameTransformerCfg`）；`FrankaCameraCfg(wrist_cam)`；`FrankaMimicEnv(ManagerBasedRLMimicEnv)` 实现 mimic 五件套（`get_robot_eef_pose`/`target_eef_pose_to_action`/`action_to_target_eef_pose`/`actions_to_gripper_actions`/`get_object_poses`）；`_franka_robot_cfg_on_stand()` 用 `compose_on_stand_usd` 合成台座 USD。
- **G1**（`g1/g1.py`，988 行）：`G1WBCJointEmbodiment`/`G1WBCPinkEmbodiment`/`G1WBCAgilePinkEmbodiment`/`G1WBCAgileJointEmbodiment`；动作 = `G1DecoupledWBC*ActionCfg`（WBC 下半身 + 直接关节或 PINK IK 上半身，来自 `isaaclab_arena_g1` 扩展）；`G1_CFG`/`G1_AGILE_CFG`（按二阶 PD 模型重算增益对齐 AGILE 循环策略）；`G1MimicEnv` 动作布局 17D（双夹爪+双腕 pos/quat+body 命令），物体位姿在**骨盆系**；`get_navigation_state`。
- **GR1T2**（`gr1t2/gr1t2.py`）：`GR1T2JointEmbodiment`（36 关节位置，`GR1T2HighPDSceneCfg` 高 PD + 锁腰）、`GR1T2PinkEmbodiment`（`upper_body_ik` PINK IK，USD→URDF）；`GR1T2MimicEnv` 动作 36D = 双 EEF(3+4)×2 + 双夹爪(11)×2。
- **Galbot**（`galbot/galbot.py`）：仅左臂（右臂吸盘 NotImplemented）；`RMPFlowActionCfg`（`GALBOT_LEFT_ARM_RMPFLOW_CFG`，机座系 EEF 观察）；事件随机化 std=0（RMPFlow 要求初始关节一致）。
- **Droid**（`droid/droid.py` + `actions.py` + `observations.py`）：复刻 DROID 数据采集硬件；三种动作变体（`droid_differential_ik`/`droid_rel_joint_pos`/`droid_abs_joint_pos`）；`BinaryJointPositionZeroToOneAction` 兼容 DROID 0-1 夹爪标签；观测归一化到 0-1；`stand_height_m=1.35` 可调台座；无 reward/mimic（数据模式）。
- **Agibot**（`agibot/agibot.py`）：左/右臂可选（`ArmMode`），RMPFlow；`AgibotMimicEnv(FrankaMimicEnv)` 仅覆写 `get_object_poses`（机座系物体位姿）。
- **Kuka Allegro**（`kuka_allegro/kuka_allegro.py`）：灵巧操作（dexsuite 兼容）；4 个指尖 `ContactSensorCfg`；`KukaAllegroStateObservationCfg(dexsuite.ObservationsCfg)` 加 `proprio.contact`；相对关节位置动作；无相机（断言禁止）。
- **`no_embodiment.py`**：`NoEmbodiment(EmbodimentBase)` + `EmptyActionsCfg`——无机器人环境占位。
- **`robot_on_stand_utils.py`**：`RobotPrimSpec`/`StandPrimSpec` 数据类 + `compose_on_stand_usd()`（`functools.cache`，把台座挂在机器人 base link 下，`UsdGeom.BBoxCache` 量测缩放对齐，写 `~/.cache/isaaclab_arena/usd/robot_on_stand/`）。
- **`common/arm_mode.py`**：`ArmMode`（`SINGLE_ARM`/`DUAL_ARM`/`LEFT`/`RIGHT`，`get_other_arm()`）。
- **`common/mimic_utils.py`**：`get_rigid_and_articulated_object_poses(state, env_ids)`（刚体/关节物体 4×4 位姿矩阵提取，被 Franka/GR1T2 mimic 复用）。

### 5.3 与 Scene / 环境构建的交互

`ArenaEnvBuilder` 通过 `embodiment.get_scene_cfg()` 得到 robot ArticulationCfg（prim_path `{ENV_REGEX_NS}/Robot`，scene key `"robot"`），与 Scene/Task 的 cfg 合并；动作/观察/奖励/事件/记录器配置全部经 `get_*_cfg()` 系列接口被 builder 拼装；`get_xr_cfg()` 供遥操作设备装配；`get_mimic_env()` 供 mimic 模式注册环境入口。

---

## 6. `terms/`、`teleop/`、`utils/` 模块

### 6.1 `terms/` — Isaac Lab manager term 函数库

把 Arena 的 Pose/关节/放置语义翻译成 `func(env, asset_cfg, ...)` 签名，供 observation 与 reset 事件使用：
- **`articulations.py`**：`joint_acc(env, asset_cfg)`——关节加速度观测。
- **`transforms.py`**：`transform_pose_from_world_to_target_frame()`（目标 link 在目标 frame 系的位姿，G1 手腕→骨盆用）、`get_target_link_position/quaternion_in_target_frame()`、`get_navigate_cmd()`（读 G1 导航命令）、`get_asset_position()/get_asset_quaternion()`。
- **`events.py`**：`set_object_pose()` / `set_object_pose_per_env()`（写 root pose + 速度，env 原点偏移）、`reset_placement_asset_pose()` / `reset_placement_asset_pose_per_env()`（embodiment 复合资产复位，支持辅助 prim）、`reset_all_articulation_joints()`（回到 default root/joint state）。

### 6.2 `teleop/` — OpenXR 遥操作

- **`single_arm_openxr_pipeline.py`**：`build_single_arm_openxr_pipeline()`——用 isaacteleop 把右手控制器位姿+触发键重定向为 **7D 相对 IK 动作**：`ControllersSource`+`HandsSource` → `Se3RelRetargeter`（右手增量位姿）+ `GripperRetargeter`（触发阈值 0.5）→ `TensorReorderer` 拼成 `[dx,dy,dz,rx,ry,rz,gripper]` → `OutputCombiner`。被 `retargeter_library.FrankaIKIsaacTeleopRetargeter` 引用。
- **`cli.py`**：`enable_openxr_teleop_from_cli(args_cli)`——`--xr` 与 `--teleop_device openxr` 的 CLI 归一化。

### 6.3 `utils/` — 工具库（约 20 个模块）

- **configclass.py**（组合架构核心）：`make_configclass()`（动态生成 configclass）、`combine_configclasses()` / `combine_configclass_instances()`（把 Scene/Embodiment/Task 的 cfg 实例**按字段合并**成单一 configclass，类型冲突报错，`combine_post_inits` 串联 `__post_init__`）、`transform_configclass_instance()`、`check_configclass_field_duplicates()`。
- **pose.py**：`Pose`（position_xyz + rotation_xyzw）、`compose_poses`、`PosePerEnv`（每 env 独立位姿）、`PoseRange`（随机化区间）——三种位姿类型贯穿资产/放置/重置全链路。
- **bounding_box.py**：`AxisAlignedBoundingBox`（含 per-env 张量化 min/max、`get_corners_at`、90° Z 旋转）、`quaternion_to_90_deg_z_quarters`。
- **cameras.py**：`ArenaCameraCfg`（embodiment 相机配置基类）、`make_camera_observation_cfg()`、`get_viewer_cfg_look_at_object()`。
- **usd_helpers.py**（421 行）：USD 工具——`is_rigid_body`/`is_articulation_root`/`object_type_for_prim`、`compute_local_bounding_box_from_usd/prim`、`extract_trimesh_from_usd_path`（碰撞网格）、`open_stage`、`has_light`、`articulation_joint_names`。
- **usd_prim_tree.py**：`load_usd_prim_tree()`（USD prim 树记录）；**usd/rigid_bodies.py**：`find_shallowest_rigid_body()`；**usd/object_set_utils.py**：`rescale_rename_rigid_body_and_save_to_cache()`。
- **physics_settle.py**：`step_physics()`、`are_all_objects_settled_per_env()`（物理沉降判定）。
- **joint_utils.py**：归一化/反归一化关节读写（affordances 依赖）。
- 其他：`velocity.py`（`Velocity`）、`yaw.py`、`math.py`、`random.py`、`device.py`、`multiprocess.py`（`get_local_rank`）、`singleton.py`（`SingletonMeta`）、`phyx_utils.py`（`add_contact_report`）、`trimesh.py`、`hydra_overrides.py`、`reload_modules.py`、`isaac_sim_debug_draw.py`；子包 `isaaclab_utils/`：`simulation_app.py`（`SimulationAppContext`、`get_app_launcher`、`teardown_simulation_app`、`reapply_viewer_cfg`）、`recorders.py`（相机/策略动作记录器 cfg）、`isaac_rtx_renderer_patch.py`；`usd_pose_helpers.py`。

---

## 7. 环境组合架构：Scene / Embodiment / Task → 环境

### 7.1 三种原语的接口约定

| 原语 | 基类 | 核心产出（供 builder 合并） |
|---|---|---|
| **Scene** | `scene/scene.py:Scene` | `get_scene_cfg()`（各资产 `(name, IsaacLabCfg)` 字段）、`get_events_cfg()`（资产 reset 事件）、`get_observation_cfg()` 等槽位、`get_objects_with_relations()`（供关系求解）、`get_asset_variations()` |
| **Embodiment** | `embodiments/embodiment_base.py:EmbodimentBase` | `get_scene_cfg()`（robot ArticulationCfg + 相机）、`get_action_cfg()`、`get_observation_cfg()`、`get_events_cfg()`、`get_rewards_cfg()`、`get_curriculum_cfg()`、`get_commands_cfg()`、`get_termination_cfg()`、`get_recorder_term_cfg()`、`get_xr_cfg()`、`get_mimic_env()`、`get_teleop_target_frame_prim_path()` |
| **Task** | `tasks/task_base.py:TaskBase` | `get_scene_cfg()`、`get_termination_cfg()`、`get_events_cfg()`、`get_mimic_env_cfg(arm_mode)`、`get_metrics()`、`get_observation_cfg()`/`get_rewards_cfg()`/`get_curriculum_cfg()`/`get_commands_cfg()`（可选）、`get_viewer_cfg()`、`get_episode_length_s()`、`get_progress_objectives()`、`apply_reachability_constraints()`、`success_state_transition()` |

Task 具体实现（`tasks/`，均 `@register_task` 注册进 `TaskRegistry`）：`LiftObjectTask`、`PickAndPlaceTask`、`OpenDoorTask`、`CloseDoorTask`、`PressButtonTask`、`TurnKnobTask`、`RotateRevoluteJointTask`、`SortingTask`、`AssemblyTask`、`GoalPoseTask`、`PlaceUprightTask`、`NoTask`；复合任务 `CompositeTaskBase`（subtasks 无序组合 + `SubtaskSuccessRateMetric`）/ `SequentialTaskBase`（顺序）。

### 7.2 组装容器 — `IsaacLabArenaEnvironment`（`environments/isaaclab_arena_environment.py`）

- 字段：`name`、`scene: Scene`、`embodiment: EmbodimentBase | None`、`task: TaskBase | None`、`teleop_device`、`env_cfg_callback`（环境 cfg 后处理回调）、`rl_framework_entry_point`/`rl_policy_cfg`（RL 框架注册键，如 `rsl_rl_cfg_entry_point`）、`episode_recorder_terms`、`placer_params: ObjectPlacerParams`。
- 工厂契约：`arena_environment_factory.py` 的 `ArenaEnvironmentFactory.build(cfg) -> IsaacLabArenaEnvironment`，子类按名称注册进 `EnvironmentRegistry`。

### 7.3 编译 — `ArenaEnvBuilder`（`environments/arena_env_builder.py`，507 行）

`compose_manager_cfg()` 流水线：
1. **关系求解**（`cfg.solve_relations=True` 时）：`_solve_relations()` → `task.apply_reachability_constraints()` → 收集 `scene.get_objects_with_relations()` + embodiment（若有关系）→ `solve_and_apply_relation_placement()`（`environments/relation_solver_interface.py`：构造 `PooledObjectPlacer`，`resolve_on_reset=True` 时注册 `placement_reset` reset 事件，False 时写固定 `PosePerEnv`）。
2. **变体注入**：Hydra overrides → `VariationRecorder.attach()` → `_apply_build_time_variations()`（构造期变异，须在 scene_cfg 物化前）。
3. **cfg 合并**（核心，全部用 `combine_configclass_instances`）：
   - `scene_cfg = Scene(InteractiveSceneCfg(num_envs, env_spacing, replicate_physics=False), scene.get_scene_cfg(), embodiment.get_scene_cfg(), task.get_scene_cfg())`
   - `observation_cfg` = scene + embodiment + task；`events_cfg` = embodiment + scene + task + placement 事件 + variations 事件 + progress 事件；`termination_cfg` = task + scene + embodiment；`actions_cfg` = embodiment；`recorder_manager_cfg` = metrics + task + embodiment + progress；`rewards/curriculum/commands_cfg` 同理三方合并；`episode_recorders_cfg` 内置 core/variations/progress 三项。
4. **设备/遥操作**：`DeviceRegistry.get_teleop_device_cfg(teleop_device, embodiment)` → `IsaacTeleopCfg` 或 `DevicesCfg`。
5. **产出**：`IsaacLabArenaManagerBasedRLEnvCfg`（非 mimic）或 `IsaacArenaManagerBasedMimicEnvCfg`（mimic 模式，走 `task.get_mimic_env_cfg(arm_mode)`）；`env_cfg.seed`、物理后端 presets（`ArenaPhysicsCfg`：`physx`/`newton`）在此设置；`env_cfg_callback` 最后改写。
6. `build_registered()`：`gym.register(id=name, entry_point=..., kwargs={"env_cfg_entry_point": env_cfg, [rl键]: policy_cfg})` → `parse_env_cfg()` → 返回 `(name, cfg, env_kwargs)`；`make_registered()` 再 `gym.make()` 并 `reapply_viewer_cfg(env)`。

### 7.4 运行时环境 — `IsaacLabArenaManagerBasedRLEnv`

- 继承 `isaaclab.envs.ManagerBasedRLEnv`；`load_managers()` 追加 `MetricsManager`（`cfg.metrics`）与 `EpisodeRecorderManager`（`cfg.episode_recorders`）；`_reset_idx()` 覆写：先 `record_pre_reset` 再推进 episode 计数再 super；`compute_metrics()` → `MetricsDataCollection`；`get_language_instruction()`（`cfg.task_description`，即自然语言任务指令）；`variation_recorder`/`object_initial_rest_pose_recorder` 属性。
- 配置 `IsaacLabArenaManagerBasedRLEnvCfg`：`commands/rewards/curriculum = None`（Arena 用任务层配置）、`sim.dt=1/200`、`decimation=4`、`task_description`。
- Isaac Lab 训练互操作：`isaaclab_interop.environment_registration_callback` 作为 `--external_callback` 传入 Isaac Lab 训练脚本（`--task <env_name>`），启动 SimulationApp 后从 `EnvironmentRegistry` 取工厂、从 CLI 建环境、`build_registered()` 注册，返回剩余参数给 Isaac Lab。

### 7.5 声明式入口 — `environment_spec/`（YAML 环境图）

- `ArenaEnvGraphSpec`（pydantic）：`env_name`、`embodiment: AssetSpec`、`background: AssetSpec`、`objects: [AssetSpec]`、`object_references`、`relations: [SpatialRelationSpec]`、`placement_validators`、`task: CompositeTaskSpec`（composition ∈ atomic/parallel/sequential + subtasks）、`cli_override_specs`；`from_yaml()/from_dict()/write_yaml()/apply_cli_override_args()/to_arena_env()`。
- `arena_env_graph_conversion_utils.build_arena_env_from_graph_spec()`：从 `AssetRegistry` 实例化各节点资产（`registry_name` → 类 → 实例）→ 检查/注入默认光源 → `_attach_spatial_relations_to_assets()`（`ObjectRelationLibraryRegistry` 按 kind 实例化关系并 `add_relation`）→ 组 `IsaacLabArenaEnvironment(scene=Scene(非 embodiment 资产), embodiment=..., task=build_task_from_spec(...), placer_params=build_checks_for_placer_params(...))`。
- 意义：**一条 YAML 即可声明"机器人 + 背景 + 物体 + 空间关系 + 复合任务"完整环境**，是"可组合环境创建"的声明式最高层。

### 7.6 一张图总结

```
YAML (ArenaEnvGraphSpec)
  └─ to_arena_env() ──► IsaacLabArenaEnvironment
                          ├─ scene: Scene { objects / backgrounds / object_references / 关系 }
                          ├─ embodiment: EmbodimentBase (robot 资产 + 动作/观察/奖励/相机/XR 配置)
                          ├─ task: TaskBase (终止/事件/指标/mimic/进度目标/可达性)
                          ├─ teleop_device / env_cfg_callback / placer_params
                          └─ ArenaEnvBuilder.compose_manager_cfg()
                               ├─ 关系求解 → placement_reset 事件（PooledObjectPlacer 布局池）
                               ├─ 变体注入（VariationRecorder）
                               └─ combine_configclass_instances 合并
                                    scene / observations / events / terminations /
                                    actions / rewards / curriculum / commands /
                                    recorders / metrics / episode_recorders
                                    → IsaacLabArenaManagerBasedRLEnvCfg
                               └─ gym.register + parse_env_cfg
                                    → IsaacLabArenaManagerBasedRLEnv（ManagerBasedRLEnv 扩展）
```

**一句话总结**：Scene 提供"世界有什么"（资产与几何关系），Embodiment 提供"谁在动、怎么控制"（机器人及其动作/观察/传感器栈），Task 提供"要达成什么"（终止/奖励/指标/进度）；`ArenaEnvBuilder` 用 `combine_configclass_instances` 把三者各自的 `get_*_cfg()` 按字段合并成一份 Isaac Lab ManagerBased 环境配置并注册进 gym——环境 = 三种原语的配置并集。
