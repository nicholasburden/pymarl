./run.sh $1 python3 src/main.py --config=reptile_3m_5m --env-config=sc2 with buffer_cpu_only=True label=reptile_3m_5m_nick checkpoint_path=results/models/reptile_3m_5m__2020-03-24_23-43-09__vdn_sc2_3m load_step=0 & ./run.sh $2 python3 src/main.py --config=reptile_3m_5m --env-config=sc2 with buffer_cpu_only=True label=reptile_3m_5m_nick checkpoint_path=results/models/reptile_3m_5m__2020-03-24_23-43-09__vdn_sc2_3m load_step=0 & ./run.sh $3 python3 src/main.py --config=reptile_3m_5m --env-config=sc2 with buffer_cpu_only=True label=reptile_3m_5m_nick checkpoint_path=results/models/reptile_3m_5m__2020-03-24_23-43-09__vdn_sc2_3m load_step=0
