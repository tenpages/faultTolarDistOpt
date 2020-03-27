## faultTolarDistOpt

### Dependencies
Run using Python 3 with numpy, torch, torchvision, blosc and mpi4py.

### Launch

Run 
`$ mpirun -n 9 python distributed_nn.py --network FC --approach baseline --worker-fail 2 --compress-grad compress --epochs 1 --max-steps 100`

more to add in the future
