# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural library furniture and rigid books for the manipulation example."""

from dataclasses import dataclass

SURFACE_Z = 0.75
BOOK_SIZE = (0.035, 0.16, 0.22)


@dataclass(frozen=True)
class Book:
    """A book's initial and final center positions in world coordinates."""

    name: str
    source: tuple[float, float, float]
    target: tuple[float, float, float]
    color: tuple[float, float, float]
    catalog_id: str
    shelf_slot: str
    size: tuple[float, float, float] = BOOK_SIZE
    mass: float = 0.18


BOOKS = (
    Book("return_top", (0.40, -1.55, 0.86), (0.40, 1.50, 0.86), (0.65, 0.12, 0.09), "LIB-001", "A-01"),
    Book("return_bottom", (0.60, -1.55, 0.86), (0.60, 1.50, 0.86), (0.12, 0.32, 0.65), "LIB-002", "A-02"),
    Book("retrieve", (0.80, 1.50, 0.86), (0.55, -0.70, 0.86), (0.12, 0.52, 0.26), "LIB-003", "A-03"),
)

CATALOG = {
    book.catalog_id: {
        "name": book.name,
        "shelf_slot": book.shelf_slot,
        "shelf_position": book.source if book.name == "retrieve" else book.target,
    }
    for book in BOOKS
}


def create_room(sim):
    """Spawn the room and furniture and return the three movable rigid books.

    Args:
        sim: An initialized Isaac Lab SimulationContext before its first reset.

    Returns:
        Rigid books indexed by the names in BOOKS.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    def box(name, size, position, color, collision=True):
        cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.72),
        )
        cfg.func(f"/World/Library/{name}", cfg, translation=position)

    wood = (0.39, 0.23, 0.12)
    wall = (0.80, 0.82, 0.76)
    box("Floor", (6.0, 6.0, 0.12), (0, 0, -0.06), (0.55, 0.48, 0.36))
    # Leave the camera-facing side open, and a door-size gap in the left wall.
    box("RearWall", (6.0, 0.12, 2.8), (0, 3.0, 1.4), wall)
    box("RightWall", (0.12, 6.0, 2.8), (3.0, 0, 1.4), wall)
    box("LeftWallRear", (0.12, 3.8, 2.8), (-3.0, 1.1, 1.4), wall)
    box("LeftWallFront", (0.12, 0.8, 2.8), (-3.0, -2.6, 1.4), wall)

    def bookshelf(name, x, y):
        width, depth, height = 1.15, 0.32, 1.95
        box(f"{name}Back", (width, 0.035, height), (x, y + depth / 2, height / 2), wood)
        for side in (-1, 1):
            box(f"{name}Side{side + 1}", (0.035, depth, height), (x + side * width / 2, y, height / 2), wood)
        for level in range(5):
            z = 0.12 + level * 0.44
            box(f"{name}Shelf{level}", (width, depth, 0.035), (x, y, z), wood)
            if level == 4:
                continue
            for index in range(9):
                color = ((0.18, 0.32, 0.52), (0.55, 0.16, 0.12), (0.30, 0.45, 0.23))[index % 3]
                book_height = 0.22 + 0.025 * (index % 4)
                box(
                    f"{name}Book{level}_{index}",
                    (0.075, 0.22, book_height),
                    (x - 0.46 + index * 0.10, y - 0.015, z + 0.0175 + book_height / 2),
                    color,
                )

    for index, x in enumerate((-1.9, -0.55, 0.80, 2.15)):
        bookshelf(f"RearShelf{index}", x, 2.65)
    # Two short freestanding rows leave a central aisle around the robot station.
    bookshelf("LeftAisleShelf", -1.9, 0.75)
    bookshelf("RightAisleShelf", 2.15, 0.75)
    box("AisleRunner", (1.25, 4.1, 0.005), (0.1, 0.35, 0.003), (0.20, 0.26, 0.25), False)

    # The actual target slots occupy the middle shelf of this full-height bookcase.
    # The 1.03 m opening also clears the forearm and wrist during horizontal insertion.
    box("CatalogShelfBack", (0.935, 0.035, 2.10), (0.60, 1.82, 1.05), wood)
    for index, x in enumerate((0.15, 1.05)):
        box(f"CatalogShelfSide{index}", (0.035, 0.52, 2.10), (x, 1.56, 1.05), wood)
    for index, top in enumerate((0.15, SURFACE_Z, 1.815, 2.10)):
        box(f"CatalogShelfBoard{index}", (0.90, 0.52, 0.035), (0.60, 1.56, top - 0.0175), wood)
        if index in (1, 3):
            continue
        for item in range(8):
            color = ((0.18, 0.32, 0.52), (0.55, 0.16, 0.12), (0.30, 0.45, 0.23))[item % 3]
            box(
                f"CatalogBackgroundBook{index}_{item}",
                (0.06, 0.20, 0.25),
                (0.27 + item * 0.095, 1.56, top + 0.125),
                color,
            )
    # Colored markers identify the three catalog slots without relying on USD text.
    for index, book in enumerate(BOOKS):
        x, _, _ = CATALOG[book.catalog_id]["shelf_position"]
        box(f"SlotMarker{index}", (0.07, 0.003, 0.018), (x, 1.298, 0.727), book.color, False)

    def open_box(name, x, y, width, depth, color):
        box(f"{name}Support", (width - 0.04, depth - 0.04, 0.71), (x, y, 0.355), (0.30, 0.32, 0.31))
        box(f"{name}Bottom", (width, depth, 0.04), (x, y, SURFACE_Z - 0.02), color)
        for side in (-1, 1):
            box(
                f"{name}SideX{side + 1}",
                (0.015, depth, 0.09),
                (x + side * (width / 2 - 0.0075), y, SURFACE_Z + 0.045),
                color,
            )
            box(
                f"{name}SideY{side + 1}",
                (width, 0.015, 0.09),
                (x, y + side * (depth / 2 - 0.0075), SURFACE_Z + 0.045),
                color,
            )

    # Both boxes sit beside the mobile base's path, with their open tops below the grasp.
    open_box("ReturnBox", 0.50, -1.55, 0.38, 0.34, (0.56, 0.31, 0.12))
    open_box("LoanBox", 0.55, -0.70, 0.34, 0.34, (0.16, 0.36, 0.44))

    books = {}
    for book in BOOKS:
        cfg = sim_utils.CuboidCfg(
            size=book.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=book.mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.85, 0.74), roughness=0.75),
        )
        books[book.name] = RigidObject(
            RigidObjectCfg(
                prim_path=f"/World/Book_{book.name}",
                spawn=cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=book.source),
            )
        )

        # Covers and spine are visual children of the rigid page block.
        # Keeping these parts within the physical cuboid preserves its collision envelope.
        def book_part(name, size, position, color):
            part = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
            )
            part.func(f"/World/Book_{book.name}/{name}", part, translation=position)

        for side in (-1, 1):
            book_part(f"Cover{side + 1}", (0.002, 0.16, 0.22), (side * 0.0165, 0, 0), book.color)
        book_part("Spine", (0.035, 0.003, 0.22), (0, -0.0785, 0), book.color)
        book_part("SpineLabel", (0.024, 0.001, 0.036), (0, -0.0801, -0.045), (0.94, 0.93, 0.85))
        for stripe in range(int(book.catalog_id[-1])):
            book_part(
                f"CatalogStripe{stripe}",
                (0.004, 0.001, 0.022),
                (-0.008 + stripe * 0.008, -0.0807, -0.045),
                (0.14, 0.16, 0.18),
            )
        for line in range(5):
            book_part(
                f"PageEdge{line}",
                (0.027, 0.001, 0.0006),
                (0, 0.0801, -0.072 + line * 0.036),
                (0.63, 0.61, 0.54),
            )
    return books


def add_book_tongs(stage, ee_path, opening, paddle_length=0.14):
    """Create thin visual paddles suitable for picking a book off a flat surface."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    from isaaclab_arena_examples.bento.meal import TCP_OFFSET

    def part(path, size, position, color):
        cfg = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
        )
        cfg.func(path, cfg, translation=position)

    metal = (0.40, 0.45, 0.48)
    part(ee_path + "/LibraryToolStem", (0.025, 0.025, 0.14), (0, 0, -0.07), metal)
    part(ee_path + "/LibraryToolHousing", (0.23, 0.035, 0.035), (0, 0, -0.15), metal)
    ops = []
    for name, side in (("LibraryJawLeft", -1), ("LibraryJawRight", 1)):
        root = UsdGeom.Xform.Define(stage, ee_path + "/" + name)
        op = root.AddTranslateOp()
        op.Set(Gf.Vec3d(side * opening / 2, 0, TCP_OFFSET[2]))
        part(str(root.GetPath()) + "/Finger", (0.007, paddle_length, 0.052), (0, 0, 0.018), metal)
        part(str(root.GetPath()) + "/Pad", (0.002, paddle_length, 0.016), (-side * 0.004, 0, 0), (0.8, 0.82, 0.78))
        ops.append(op)
    return ops


def create_mobile_base(sim):
    """Create a kinematic wheeled pedestal whose root pose is on the floor.

    Args:
        sim: An initialized SimulationContext before its first reset.

    Returns:
        A RigidObject movable through write_root_pose_to_sim; the arm mount is 0.75 m above its root.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from pxr import Gf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(sim.stage, "/World/MobileBase")
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, -1.50, 0.0))
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(True)

    def part(name, size, position, color):
        cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.55),
        )
        cfg.func(f"/World/MobileBase/{name}", cfg, translation=position)

    part("Chassis", (0.50, 0.46, 0.13), (0, 0, 0.205), (0.14, 0.23, 0.31))
    part("Column", (0.20, 0.20, 0.43), (0, 0, 0.485), (0.55, 0.59, 0.61))
    part("ArmMount", (0.32, 0.30, 0.05), (0, 0, 0.725), (0.27, 0.31, 0.35))
    part("FrontPanel", (0.31, 0.007, 0.045), (0, 0.234, 0.21), (0.22, 0.67, 0.70))
    for xside in (-1, 1):
        for yside in (-1, 1):
            cfg = sim_utils.CylinderCfg(
                radius=0.09,
                height=0.04,
                axis="X",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.055, 0.065, 0.075), roughness=0.9),
            )
            cfg.func(
                f"/World/MobileBase/Wheel{xside + 1}_{yside + 1}",
                cfg,
                translation=(xside * 0.25, yside * 0.15, 0.095),
            )
    return RigidObject(
        RigidObjectCfg(
            prim_path="/World/MobileBase",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.50, 0.0)),
        )
    )
