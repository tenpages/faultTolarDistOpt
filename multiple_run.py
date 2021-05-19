import subprocess

#model_names = ['geomedian','medofmeans','cwtm','mkrum5','normfilter']
#models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum','grad_norm']
#fault_types = ['revgrad2','labelflipping','gaussian']
#model_names = ['bulyannormfilter']
#models = ['bulyan_grad_norm']
#model_names = ['mkrum5','cwtm',"medofmeans"]
#models = ['multi_krum','coor_wise_trimmed_mean','median_of_means']
model_names = ['async']
models = ['asynchronous_drop_f']
fault_types = ['async']#,'normfilter','labelflipping']
fault_names = ['async']#,'normfilter','labelflipping']
#nums_faults = [1,3,5,10,15]
nums_faults = [0]
batch_sizes = ['128']
#acc_alpha = '60'
#acc_alphas = ['20','40','60']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = ['mpirun', '-n', '21',
						'python', 'distributed_nn.py',
						'--batch-size=' + batch_size,
						'--max-steps', '1000',
						'--epochs', '100',
						'--network', 'LeNet',
						'--mode', model,
						'--dataset', 'Fashion-MNIST',
						'--approach', 'baseline',
						'--err-mode', fault_type,
						'--lr', '0.01',
						'--train-dir', 'output/models/async-fmnist/' + fault_name + '-' + str(i) + '/',
						'--accumulative', 'False',
						'--worker-fail', str(i),
						'--fault-thrshld', str(i),
						'--data-distribution', 'same',
						'--calculate-cosine', 'False',
						'--checkpoint-step', '0',
						'--eval-freq', '1',
						'--diff-privacy-param', '0']
				print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
				print(' '.join(args))
				results = subprocess.run(args, capture_output=True)
				if results.returncode==0 and results.stdout != None:
					with open('logs-async-fMNIST-LeNet-' + batch_size + '-' + model_name + '-20-' + str(i),'w') as f:
						f.write(results.stdout.decode())
					print("finished")
					print("========================")
				else:
					print(results.stderr.decode())
					print("failed")
					print("========================")

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = ['mpirun', '-n', '21',
						'python', 'distributed_nn.py',
						'--batch-size=' + batch_size,
						'--max-steps', '1000',
						'--epochs', '100',
						'--network', 'LeNet',
						'--mode', model,
						'--dataset', 'MNIST',
						'--approach', 'baseline',
						'--err-mode', fault_type,
						'--lr', '0.01',
						'--train-dir', 'output/models/async/' + fault_name + '-' + str(i) + '/',
						'--accumulative', 'False',
						'--worker-fail', str(i),
						'--fault-thrshld', str(i),
						'--data-distribution', 'same',
						'--calculate-cosine', 'False',
						'--checkpoint-step', '0',
						'--eval-freq', '1',
						'--diff-privacy-param', '0']
				print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
				print(' '.join(args))
				results = subprocess.run(args, capture_output=True)
				if results.returncode==0 and results.stdout != None:
					with open('logs-async-MNIST-LeNet-' + batch_size + '-' + model_name + '-20-' + str(i),'w') as f:
						f.write(results.stdout.decode())
					print("finished")
					print("========================")
				else:
					print(results.stderr.decode())
					print("failed")
					print("========================")

# for batch_size in batch_sizes:
# 	for fault_type, fault_name in zip(fault_types, fault_names):
# 		for model_name, model in zip(model_names, models):
# 			args = ['mpirun', '-n', '21',
# 					'python', 'distributed_nn.py',
# 					'--batch-size=' + batch_size,
# 					'--max-steps', '1000',
# 					'--epochs', '100',
# 					'--network', 'LeNet',
# 					'--mode', 'normal',
# 					'--dataset', 'Fashion-MNIST',
# 					'--approach', 'baseline',
# 					'--err-mode', fault_type,
# 					'--lr', '0.01',
# 					'--train-dir', 'output/models/async-fmnist/sync/',
# 					'--accumulative', 'False',
# 					'--worker-fail', '0',
# 					'--fault-thrshld', '0',
# 					'--data-distribution', 'same',
# 					'--calculate-cosine', 'False',
# 					'--checkpoint-step', '0',
# 					'--eval-freq', '1',
# 					'--diff-privacy-param', '0']
# 			print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
# 			print(' '.join(args))
# 			results = subprocess.run(args, capture_output=True)
# 			if results.returncode==0 and results.stdout != None:
# 				with open('logs-async-fMNIST-LeNet-' + batch_size + '-sync-20-0','w') as f:
# 					f.write(results.stdout.decode())
# 				print("finished")
# 				print("========================")
# 			else:
# 				print(results.stderr.decode())
# 				print("failed")
# 				print("========================")

print()
model_names = ['async']
models = ['asynchronous_drop_f']
fault_types = ['async']#,'normfilter','labelflipping']
fault_names = ['async']#,'normfilter','labelflipping']
#nums_faults = [1,3,5,10,15]
nums_faults = [0]
batch_sizes = ['128']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/async-fmnist/'+fault_name+'-'+str(i)\
						+'/ --dataset Fashion-MNIST --network LeNet --eval-freq 1 > "results-async-fMNIST-LeNet-'+batch_size+'-'+fault_name+'-20-'+str(i)+'" 2>&1 &'
				print("Now evaluating "+fault_name+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/async/'+fault_name+'-'+str(i)\
						+'/ --dataset MNIST --network LeNet --eval-freq 1 > "results-async-MNIST-LeNet-'+batch_size+'-'+fault_name+'-20-'+str(i)+'" 2>&1 &'
				print("Now evaluating "+fault_name+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)

# for batch_size in batch_sizes:
# 	for fault_type, fault_name in zip(fault_types, fault_names):
# 		for model_name, model in zip(model_names, models):
# 			args = 'python distributed_eval.py --model-dir output/models/async-fmnist/sync'\
# 					+'/ --dataset Fashion-MNIST --network LeNet --eval-freq 1 > "results-async-fMNIST-LeNet-'+batch_size+'-sync-20-0" 2>&1 &'
# 			print("Now evaluating "+fault_name+" using "+model_name+" using command:")
# 			print(args)
# 			results = subprocess.run(args, shell=True)
