./run.sh $1 python3 src/main.py --config=qmix_smac --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=convddpg_input_grid_shallow mixer=qmix action_input_representation=Grid env_args.map_name=3s5z env_args.obs_decoder=grid_metamix__3s5z env_args.obs_grid_shape=12x12 label=non_rnn_conv_input_grid_12x12_qmix_3s5z batch_size_run=8 batch_size=12 & ./run.sh $2 python3 src/main.py --config=qmix_smac --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=convddpg_input_grid_shallow mixer=qmix action_input_representation=Grid env_args.map_name=3s5z env_args.obs_decoder=grid_metamix__3s5z env_args.obs_grid_shape=12x12 label=non_rnn_conv_input_grid_12x12_qmix_3s5z batch_size_run=8 batch_size=12 & ./run.sh $3 python3 src/main.py --config=qmix_smac --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=convddpg_input_grid_shallow mixer=qmix action_input_representation=Grid env_args.map_name=3s5z env_args.obs_decoder=grid_metamix__3s5z env_args.obs_grid_shape=12x12 label=non_rnn_conv_input_grid_12x12_qmix_3s5z batch_size_run=8 batch_size=12 & ./run.sh $4 python3 src/main.py --config=qmix_smac --env-config=sc2 with t_max=10000000 test_interval=20000 test_nepisode=24 test_greedy=True env_args.obs_own_health=True save_model=True save_model_interval=2000000 test_interval=20000 log_interval=20000 runner_log_interval=20000 learner_log_interval=20000 buffer_cpu_only=True agent=convddpg_input_grid_shallow mixer=qmix action_input_representation=Grid env_args.map_name=3s5z env_args.obs_decoder=grid_metamix__3s5z env_args.obs_grid_shape=12x12 label=non_rnn_conv_input_grid_12x12_qmix_3s5z batch_size_run=8 batch_size=12
