./run.sh $1 python3 src/main.py --config=vdn_journal --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=rnn_convddpg_input_grid_no_id mixer=vdn action_input_representation=Grid env_args.map_name=3m env_args.obs_decoder=grid_metamix__3m_noid_nolastaction env_args.obs_grid_shape=12x12 label=conv_ddpg_input_grid_no_id_2_vdn_3m_12x12 & ./run.sh $2 python3 src/main.py --config=vdn_journal --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=rnn_convddpg_input_grid_no_id mixer=vdn action_input_representation=Grid env_args.map_name=3m env_args.obs_decoder=grid_metamix__3m_noid_nolastaction env_args.obs_grid_shape=12x12 label=conv_ddpg_input_grid_no_id_2_vdn_3m_12x12 & ./run.sh $3 python3 src/main.py --config=vdn_journal --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=rnn_convddpg_input_grid_no_id mixer=vdn action_input_representation=Grid env_args.map_name=3m env_args.obs_decoder=grid_metamix__3m_noid_nolastaction env_args.obs_grid_shape=12x12 label=conv_ddpg_input_grid_no_id_2_vdn_3m_12x12

