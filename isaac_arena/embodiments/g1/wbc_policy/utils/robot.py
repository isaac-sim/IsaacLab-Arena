# TODO: This class and associated configs should be deleted and instead use the robot_model package
class Robot:
    def __init__(self, config):
        self.ROBOT_TYPE = config["ROBOT_TYPE"]
        self.NUM_JOINTS = config["NUM_JOINTS"]