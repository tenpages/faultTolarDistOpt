# fault-check rate = .2
# Byzantine worker fault rate = .2
# 
#
mkdir output
mpirun -n 11 --mca btl_base_warn_component_unused 0 python3.8 linreg_test.py --batch-size=10 --max-steps 300 --epochs 1 --network LinregTest --mode normal --dataset LinregTest --data-distribution same --approach baseline --train-dir output/ --worker-fail 1 --err-mode rev_grad --eval-freq 1 --redundancy --q 0.2 --p 0.2 > logs-linreg-ex &
