from onpolicy.envs.crowd_sim.utils.state import EntityState
from onpolicy.envs.crowd_sim.utils.state import ObservableState, FullState, AgentState

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # position
        self.px = 0
        self.py = 0
        # speed
        self.vx = 0
        self.vy = 0
        self.v_pref = 0
        # theate
        self.theta = 0
        # radius
        self.radius = 0
        # name 
        self.id = -1
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # color
        self.color = None
        # max speed and accel  
        self.max_speed = None
        self.accel = None
        self.visible = None
        # state
        self.state = EntityState()

    def set(self, px, py, vx, vy, radius=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy

        if radius is not None:
            self.radius = radius    

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_observable_state_list_noV(self):
        return [self.px, self.py, self.radius]


    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_position(self):
        return self.px, self.py

    def get_velocity(self):
        return self.vx, self.vy


    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]
