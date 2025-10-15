python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf20-91/ --pem_num_particles 8 --device 'cuda:0' --batch_size 1024
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf20-91/ --pem_num_particles 128 --device 'cuda:0' --batch_size 128
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf20-91/ --pem_num_particles 1024 --device 'cuda:0' --batch_size 8
python -u -m comp_reasoning.eval_3sat --path comp_reasoning/data/3sat/test/uf20-91/ --pem_num_particles 2048 --device 'cuda:0' --batch_size 6
