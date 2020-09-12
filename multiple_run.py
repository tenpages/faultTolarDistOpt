import subprocess

#model_names = ['geomedian','medofmeans','cwtm','mkrum5']
#models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum']
#fault_types = ['revgrad2','labelflipping','gaussian']
model_names = ['normfilter']
models = ['grad_norm']
fault_types = ['normfilter']
nums_faults = [2,6,8,10,12]
batch_sizes = ['32','128','256']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type in fault_types:
			for model_name, model in zip(model_names, models):
				args = ['mpirun', '-n', '41', 
						'python', 'distributed_nn.py', 
						'--batch-size=' + batch_size, 
						'--max-steps', '500', 
						'--epochs', '100', 
						'--network', 'LeNet', 
						'--mode', model, '--multi-krum-m','5',
						'--dataset', 'CIFAR10', 
						'--approach', 'baseline', 
						'--err-mode', fault_type, 
						'--lr', '0.01', 
						'--train-dir', 'output/models/paper2/CIFAR-LeNet/' + batch_size + '/' + fault_type + '/' + model_name + '/40-' + str(i) + '/', 
						'--accumulative', 'False', 
						'--worker-fail', str(i), 
						'--fault-thrshld', str(i), 
						'--data-distribution', 'same', 
						'--calculate-cosine', 'False', 
						'--checkpoint-step', '0', 
						'--eval-freq', '1']
				print("Now running experiments on "+fault_type+" using "+model_name+" using command:")
				print(' '.join(args))
				results = subprocess.run(args, capture_output=True)
				if results.returncode==0 and results.stdout != None:
					with open('logs-paper2-cifar-' + batch_size + '-' + fault_type + '-' + model_name + '-40-'+str(i),'w') as f:
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
		for fault_type in fault_types:
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/paper2/CIFAR-LeNet/'+batch_size+'/'+fault_type+'/'+model_name+ \
					'/40-'+str(i)+'/ --dataset CIFAR10 --network LeNet --eval-freq 1 > results-paper2-cifar-'+batch_size+'-'+fault_type+'-'+model_name+'-40-'+str(i)+' 2>&1 &'
				print("Now evaluating "+fault_type+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)
