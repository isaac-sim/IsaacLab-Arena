import isaaclab.envs.mdp as mdp_isaac_lab
from isaac_arena.embodiments.mdp.terminations import object_in_drawer
from isaac_arena.tasks.task import TaskBase
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # TODO(cvolk): Make this config generic and move instance out.
    # time_out: TerminationTermCfg = MISSING
    # termination_terms: TerminationTermCfg = MISSING
    # success: TerminationTermCfg = MISSING
    time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    object_dropped = TerminationTermCfg(
        func=mdp_isaac_lab.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("pick_up_object")},
    )
    success = TerminationTermCfg(func=object_in_drawer)


class PickAndPlaceTaskCfg(TaskBase):
    def __init__(self, time_out_functor=None, termination_terms_functor=None, success_functor=None):
        self.time_out_functor = time_out_functor
        self.termination_terms_functor = termination_terms_functor
        self.success_functor = success_functor

    def get_termination_cfg(self):
        return TerminationsCfg()

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")
