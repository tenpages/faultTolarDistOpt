import subprocess

model_names = ['cwtm','normfilter','nofilter']
models = ['coor_wise_trimmed_mean','grad_norm','normal']
fault_types = ['rev_grad_2','gaussian']
fault_names = ['revgrad2','gaussian']
nums_faults = [1]
batch_sizes = ['1']
#acc_alphas = ['20','40','60']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = ['mpirun', '-n', '7',
						'python', 'distributed_nn.py',
						'--batch-size=' + batch_size,
						'--max-steps', '2000',
						'--epochs', '10000',
						'--network', 'FC',
						'--mode', model,
						'--dataset', 'ApproxReg4',
						'--approach', 'baseline',
						'--err-mode', fault_type,
						#'--lr', '0.01',
						'--train-dir', 'output/models/apprx9/' + fault_name + '/' + model_name + '/',
						'--worker-fail', str(i),
						'--data-distribution', 'distributed',
						'--checkpoint-step', '0',
                        '--momentum', '0',
                        '--save-honest-list', 'True',
                        '--omit-agents', 'False',
                        '--diminishing-lr', 'True',
						'--eval-freq', '1']
				print("Now running experiments fault-type "+fault_name+" using "+model_name+" using command:")
				print(' '.join(args))
				results = subprocess.run(args, capture_output=True)
				if results.returncode==0 and results.stdout != None:
					with open('logs-approx9-' + fault_name + '-' + model_name + '-2000-1','w') as f:
						f.write(results.stdout.decode())
					print("finished")
					print("========================")
				else:
					print(results.stderr.decode())
					print("failed")
					print("========================")

print()
for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/apprx9/'+fault_name+'/'+model_name \
					+'/ --dataset ApproxReg4 --network FC --eval-freq 1 > results-approx9-'+fault_name+'-'+model_name+'-2000-1 2>&1 &'
				print("Now evaluating "+fault_name+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)
