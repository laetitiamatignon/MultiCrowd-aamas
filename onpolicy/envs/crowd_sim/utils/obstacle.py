import numpy as np
from numpy.linalg import norm
import abc
import logging
from onpolicy.envs.crowd_sim.crowd_nav.policy.policy_factory import policy_factory
from onpolicy.envs.crowd_sim.utils.action import ActionXY, ActionRot
from onpolicy.envs.crowd_sim.utils.state import ObservableState, FullState, AgentState

from onpolicy.envs.crowd_sim.utils.entity import Entity

class Obstacle(Entity):
    def __init__(self, config):
        """
        Base class for robot and human. Have the physical attributes of an agent.
        """
        super().__init__()
        param = config.obstacle
        self.visible = True
        self.v_pref = 0
        self.radius = param.max_radius

    def print_info(self):
        logging.info('Obstacle is {}'.format(
            'visible' if self.visible else 'invisible'))

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.radius = np.random.uniform(0., self.radius)
