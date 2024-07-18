#!/bin/sh
env="CROWD"
scenario="simple_circle_obstacle"
num_agents=5
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ./scripts/train/train_crowd.py --share_policy --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed 145 \
    --n_training_threads 2 --n_rollout_threads 16 --num_mini_batch 2 --episode_length 50 --data_chunk_length 50 --num_env_steps 20000000 \
    --archi_name "GHR" --ppo_epoch 5 --gain 0.0 --lr 4e-5 --critic_lr 4e-5 --user_name "yuchao" --label_entity 2
done


