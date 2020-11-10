#!/usr/bin/env bash

echo "creating cgroup to limit memory usage ($1)..."
sudo cgcreate -g memory:model-compression-evaluation
sudo cgset -r memory.limit_in_bytes="$1" model-compression-evaluation
sudo cgset -r memory.memsw.limit_in_bytes=$2 model-compression-evaluation

echo "running ./eval/parse_config_and_eval_model.py..."
command="python ./eval/parse_config_and_eval_model.py $3 $4 $5"
sudo cgexec -g memory:model-compression-evaluation $command

echo "deleting cgroup..."
sudo cgdelete -g memory:model-compression-evaluation
