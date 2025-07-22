


# @dataclass
# class SceneProperties:
#     supports_pick_and_place: bool = MISSING
    


arena_env = IsaacArenaEnv(
    embodiment=FrankaEmbodiment(),
    task=PickAndPlaceTask(),
    scene=PickAndPlaceScene(),
)



class Scene:

    def add_object_query(self, name: str, prim_path: str):
        self.objects[name] = ObjectQuery(prim_path)

    def query_object(self, name: str) -> ObjectQuery:
        return self.objects[name]

    



class KitchenScene(Scene):

    def __init__(self):

        # assert self.supports_pick_and_place()
        self.add_object_query("transported_object", "/World/objects/transported_object")
        self.add_object_query("cupboard_door", "/World/objects/cupboard_door")


        # transported_object = get_random_pick_and_place_object()
        # cupboard_door = 

        # target_object = self.add_object(
        #     name="target_object",
        #     spawn=get_random_pick_and_place_object(),
        # )
        # distination_pose = self.get_random_pick_and_place_pose()




class PickAndPlaceTask:

    def get_termination_cfg(self, scene: Scene) -> Any:
        assert scene.query_object("transported_object")
        assert scene.query_object("destination_aabb")



class OpenCupboardTask:

    def get_termination_cfg(self, scene: Scene) -> Any:

        cupboard = scene.get_cupboard()

