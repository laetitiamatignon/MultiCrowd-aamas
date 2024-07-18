#!/bin/sh
env="CROWD"
scenario="simple_circle_obstacle"
num_agents=5
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python3 ./scripts/render/render_crowd.py --share_policy --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed 1000 --visualize True \
    --episode_length 150 --n_training_threads 1 --n_rollout_threads 1 --data_chunk_length 2 --use_render --label_entity 2 \
    --archi_name "GHR" --model_dir "./scripts/results/NEW/VAR_K/K4" --render_episodes 400 --GHR_edge_selector_num_head 4
done

