mkdir -p ./trainlogs

#### MT10 ####

gamma=0.01
wlr=0.025

export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=famo_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=$seed replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.gamma=$gamma agent.builder.w_lr=$wlr > trainlogs/mt10_famo_gamma$gamma\_wlr$wlr\_sd$seed.log 2>&1 &

