"""Arena's MuJoCo-Warp integration for authored USD physics attributes."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo


class NewtonArenaMJWarpManager(NewtonMJWarpManager):
    """Prepare every Newton builder to ingest MuJoCo USD attributes.

    Newton only imports custom USD attributes that are registered on the
    ``ModelBuilder`` before ``add_usd``.  Isaac Lab calls this solver hook from
    ``create_builder``, so registering here covers replicated source builders,
    global builders, and the non-replicated fallback without asset-specific
    callbacks.
    """

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        super()._register_builder_attributes(builder)
        if not builder.has_custom_attribute("mujoco:condim"):
            SolverMuJoCo.register_custom_attributes(builder)


@configclass
class ArenaMJWarpSolverCfg(MJWarpSolverCfg):
    """MuJoCo-Warp configuration using Arena's USD-aware manager."""

    class_type: type[NewtonMJWarpManager] | str = NewtonArenaMJWarpManager
