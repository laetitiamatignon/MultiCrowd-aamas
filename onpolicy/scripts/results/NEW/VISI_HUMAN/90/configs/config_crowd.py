import numpy as np

class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):

    training = BaseConfig()

    # general configs for OpenAI gym env
    env = BaseConfig()
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    # if randomize human behaviors, set to True, else set to False
    env.randomize_attributes = False
    # record robot states and actions an episode for system identification in sim2real
    env.record = False
    env.load_act = False

    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 0
    reward.collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99

    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 6 * np.sqrt(2)
    sim.arena_size = 6
    sim.human_num = 17
    sim.human_num_range = 0
    sim.num_cameras = 0
    sim.obstacles = []
    # sim.obstacles = [[-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5]]
    # sim.obstacles = [[-0.25, -0.25, -0.5, -0.5], [-0.25, 0.25, -0.5, 0.5], [0.25, 0.25, 0.5, 0.5], [0.25, -0.25, 0.5, -0.5]]
    sim.width_door = 1.
    sim.width_door_interval = 0.
    sim.width_wave = 2
    sim.can_communicate = False
    sim.comm_cam_dist = 8

    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'const_vel'
    # render the simulation during training or not
    sim.render = False

    # for save_traj only
    render_traj = False
    save_slides = False
    save_path = None

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # human config
    human = BaseConfig()
    human.visible = True
    # orca or social_force for now
    human.policy = "orca"
    human.randomize_policy_parameter = True
    human.radius = 0.3
    human.radius_interval = 0.1
    human.v_pref = 1
    human.v_pref_interval = 0.
    human.sensor = "coordinates"
    human.collide = True
    # FOV = this values * PI
    human.FOV = 2.
    human.sensor_range = 10
    human.randomize_policy = False
    human.randomize_attributes = True

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    human.random_goal_changing = False
    human.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    human.end_goal_changing = True
    human.end_goal_change_chance = 1.0

    # one human may have a random chance to be blind to other agents at every time step
    human.random_unobservability = False
    human.unobservable_chance = 0.3

    human.random_policy_changing = False




    # human config
    cam = BaseConfig()
    cam.visible = False
    # orca or social_force for now
    cam.policy = "social_force"
    cam.randomize_policy = True
    cam.randomize_policy_parameter = True
    cam.radius = 0.4
    cam.radius_interval = 0.1
    cam.v_pref = 0
    cam.v_pref_interval = 0
    cam.sensor = "coordinates"
    cam.collide = False
    # FOV = this values * PI
    cam.FOV = 2.
    cam.sensor_range = 5
    cam.randomize_policy = False
    cam.randomize_attributes = True

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    cam.random_goal_changing = False
    cam.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    cam.end_goal_changing = True
    cam.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    cam.random_radii = False
    cam.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    cam.random_unobservability = False
    cam.unobservable_chance = 0.3

    cam.random_policy_changing = False

    obstacle = BaseConfig()
    obstacle.visible = True
    obstacle.v_pref = 0
    obstacle.max_radius = 0.4

    # robot config
    robot = BaseConfig()
    # whether robot is visible to all entities
    robot.visible = True
    # For baseline: srnn; our method: selfAttn_merge_srnn
    robot.policy = 'selfAttn_merge_srnn'
    robot.randomize_policy = False
    robot.randomize_policy_parameter = False
    robot.radius = 0.3
    robot.radius_interval = 0.1
    robot.v_pref = 1
    robot.v_pref_interval = 0.5
    robot.sensor = "coordinates"
    robot.collide = True
    # FOV = this values * PI
    robot.FOV = 2.
    # radius of perception range
    robot.sensor_range = 5
    robot.sensor_range_robot = 7
    robot.sensor_range_human = 5
    # robot is visible by humans
    robot.robot_in_traj_human = False
    # robot is visible by other robots
    robot.robot_in_traj_robot = True
    robot.randomize_policy = False
    robot.randomize_attributes = True
    robot.kinematics = 'holonomic'

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.neighbor_dist_interval = 0
    orca.safety_space = 0.15
    orca.safety_space_interval = 0.
    orca.time_horizon = 5
    orca.time_horizon_interval = 0
    orca.time_horizon_obst = 5
    orca.time_horizon_obst_interval = 0

    # config for social force
    sf = BaseConfig()
    sf.A = 3.
    sf.A_interval = 0.
    sf.B = 1.
    sf.B_interval = 0.
    sf.KI = 1
    sf.KI_interval = 0.

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    # data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 0.25

    # config for the GST predictor
    pred = BaseConfig()
    # see 'gst_updated/results/README.md' for how to set this variable
    # If randomized humans: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
    # else: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj
    pred.model_dir = 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'

    # LIDAR config
    lidar = BaseConfig()
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 5
    # range in meters
    lidar.range = 10

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = True
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")
