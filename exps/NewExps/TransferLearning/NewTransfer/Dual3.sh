./run.sh $1 python3 src/main.py --config=reptile_2s3z_3s5z --env-config=sc2 with buffer_cpu_only=True label=reptile_2s3z_3s5z_nick2 batch_size=12 & ./run.sh $2 python3 src/main.py --config=reptile_2s3z_3s5z --env-config=sc2 with buffer_cpu_only=True label=reptile_2s3z_3s5z_nick2 batch_size=12 & ./run.sh $3 python3 src/main.py --config=reptile_2s3z_3s5z --env-config=sc2 with buffer_cpu_only=True label=reptile_2s3z_3s5z_nick2 batch_size=12
