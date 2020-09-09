import subprocess

model_names = ['geomedian','medofmeans','cwtm','mkrum5','normfilter']
models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum','grad_norm']
fault_types = ['revgrad2','normfilter','labelflipping','gaussian']

"""
for fault_type in fault_types:
	for model_name, model in zip(model_names, models):
		args = ['mpirun', '-n', '41', 
				'python', 'distributed_nn.py', 
				'--batch-size=64', 
				'--max-steps', '700', 
				'--epochs', '100', 
				'--network', 'LeNet', 
				'--mode', model, '--multi-krum-m','5',
				'--dataset', 'CIFAR10', 
				'--approach', 'baseline', 
				'--err-mode', fault_type, 
				'--lr', '0.01', 
				'--train-dir', 'output/models/paper2/CIFAR-LeNet/64/' + fault_type + '/' + model_name + '/40-4/', 
				'--accumulative', 'False', 
				'--worker-fail', '4', 
				'--fault-thrshld', '4', 
				'--data-distribution', 'same', 
				'--calculate-cosine', 'False', 
				'--checkpoint-step', '500', 
				'--eval-freq', '1']
		print("Now running experiments on "+fault_type+" using "+model_name+" using command:")
		print(' '.join(args))
		results = subprocess.run(args, capture_output=True)
		if results.returncode==0 and results.stdout != None:
			with open('logs-paper2-cifar-64-' + fault_type + '-' + model_name + '-40-4','w') as f:
				f.write(results.stdout.decode())
			print("finished")
			print("========================")
		else:
			print("failed")
			print("========================")
"""

print()
for fault_type in fault_types:
	for model_name, model in zip(model_names, models):
		args = 'python distributed_eval.py --model-dir output/models/paper2/CIFAR-LeNet/64/'+fault_type+'/'+model_name+ \
			'/40-4/ --dataset CIFAR10 --network LeNet --eval-freq 1 --start-from 500 > results-paper2-cifar-64-'+fault_type+'-'+model_name+'-40-4 2>&1 &'
		print("Now evaluating "+fault_type+" using "+model_name+" using command:")
		print(args)
		results = subprocess.run(args, shell=True)
