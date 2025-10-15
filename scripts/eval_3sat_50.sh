#!/bin/bash
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 8 --device 'cuda:1'
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 64 --device 'cuda:1' --batch_size 50
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 128 --device 'cuda:1' --batch_size 25
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 256 --device 'cuda:1' --batch_size 20
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 512 --device 'cuda:1' --batch_size 15
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 1024 --device 'cuda:1' --batch_size 8
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf50-218/ --pem_num_particles 2048 --device 'cuda:1' --batch_size 3
