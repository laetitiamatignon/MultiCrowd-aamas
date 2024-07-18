# MultiSoc

This repository contains the code for Multi-Soc, a deep-MARL model for decentralized multi-agent learning of navigation among humans. The idea behind Multi-Soc is to exploit the graph structure of a crowd (each human is a node) and the flexibility offer by GNNs. 

## Setup
1. In a conda environment or virtual environment with Python 3.x, install the required python package
``` Bash
pip install -e .
```
2. Install Python-RVO2 library
3. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

## Organisation
### MAPPO
 - algorithms/ contains the architecture of the model.
 - Multi Crowd Nav, a multi-agent version of CrowdNav is contained in envs/ and follow the organisation of MPE. 
 - Code to perform training rollouts and policy updates are contained within the runner/ folder.
### Envs
 - Scenario contains the constraint of the agents and the crowd (observation, initial position, reward, communication) that will be used by Env
 - Core is the common mechanism of all the scenarios, whatever the conditions found in the selected scenario, it can link timestep t to timestep t+1 with step()
 - Env merges the two classes above as an interface for the model

## Usage
### Parameters
#### Simulation
All the parameters of the simulation can be found in scripts/config/config_crowd.py. Main parameters are:
* Number of humans, cameras
* Positions of obstacles (represented as broken lines) 
* Choice of the policy for entities trajectory ("social_force" or "orca" or both with randomize_policy=True )
* Radius of entites (human.radius = 0.3 and robot.radius = 0.3 for social multi-agent test and human.radius = 0.4 and robot.radius = 0.4 for others model included there)
* FOV (field of view), sensor_range (range of view), v_pref (prefered velocity)
* Parameters of ORCA and Social Force
#### Architecture
The other parameters (save, files, model dimensions, MAPPO, model choice, rendering) can be found in scripts/config/arguments.py.

### Training
Run
``` Bash
bash scripts/train_crowd_scripts/train_crowd_circle.sh
```
During training, models and logs are saved in scripts/results/scenario_name/.

### Test
Run
``` Bash
bash scripts/render_scripts/test_GHR_R3H17.sh
```
Note that the config in scripts/config/config_crowd.py will be the one used for the test.
Option --visualize presents the results like:
``` Bash
================
Navigation:
 testing success rate: 0.95
 collision rate (per agent, per episode): 0.03
 timeout rate: 0.02
 nav time: 14.33
 path length: 13.82
 average intrusion ratio: 8.94%
 average minimal distance during intrusions: 0.37
 average minimum distance to goal: 0.42
 average end distance to goal: 0.55
 average position of reached goal: 0.09,-0.14

Evaluation using 1000 episodes: mean reward 17.67544
```
and option --visualize_traj presents the simulation with rendering:
![simu_obs_cam](https://github.com/Air1Esc/MultiCrowd/assets/117373578/9a6018c8-9d8b-4913-83de-990f837b58e5)



