./run.sh $1 python3 src/main.py --config=reptile_test --env-config=sc2 with buffer_cpu_only=True use_cuda=False label=3m_and_5m_reptile2 & ./run.sh $2 python3 src/main.py --config=reptile_test --env-config=sc2 with buffer_cpu_only=True use_cuda=False label=3m_and_5m_reptile2 & ./run.sh $3 python3 src/main.py --config=reptile_test --env-config=sc2 with buffer_cpu_only=True use_cuda=False label=3m_and_5m_reptile2
