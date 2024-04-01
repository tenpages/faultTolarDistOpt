## faultTolarDistOpt

This code is built upon the existing codebase of https://github.com/hwang595/Draco.

### Dependencies

`Python 3` with `numpy`, `torch`, `torchvision`, `blosc` and `mpi4py`.

### Experiments

Run 
`$ mpirun -n 9 python distributed_nn.py --network FC --approach baseline --worker-fail 2 --compress-grad compress --epochs 1 --max-steps 100`

where `-n` controls the number of agents, `--network` is the model to use, and `--worker-fail` is the number of faulty agents to simulate. 

For more explanations and additional arguments, see comments in the file [distributed_nn.py](distributed_nn.py).
