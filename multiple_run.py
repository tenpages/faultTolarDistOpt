import subprocess

model_names = ['async-cge']
models = ['async_grad_norm']
fault_types = ['rev_grad_2','labelflipping']
fault_names = ['revgrad2','labelflipping']
nums_async = [0,1,3,5,10]
batch_sizes = ['128']

# for k in [0,1,2,3]:
# 	for batch_size in batch_sizes:
# 		for i in nums_async:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = ['mpirun', '-n', '21',
# 							'python', 'distributed_nn.py',
# 							'--batch-size=' + batch_size,
# 							'--max-steps', '1000',
# 							'--epochs', '100',
# 							'--network', 'LeNet',
# 							'--mode', model,
# 							'--dataset', 'KMNIST',
# 							'--approach', 'baseline',
# 							'--err-mode', fault_type,
# 							'--lr', '0.01',
# 							'--train-dir', 'output/models/async-ft/kmnist/20-3/' + fault_name + '/async-' + str(i) + '/run'+str(k)+'/',
# 							'--seed', str(k),
# 							'--accumulative', 'False',
# 							'--worker-fail', '3',
# 							'--fault-thrshld', '3',
# 							'--async-thrshld', str(i),
# 							'--data-distribution', 'different-dist',
# 							'--calculate-cosine', 'False',
# 							'--checkpoint-step', '0',
# 							'--eval-freq', '1',
# 							'--diff-privacy-param', '0']
# 					print("Now running experiments mpirun "+fault_name+" using "+model_name+" using command:")
# 					print(' '.join(args))
# 					results = subprocess.run(args, capture_output=True)
# 					if results.returncode==0 and results.stdout != None:
# 						with open('logs-asyncft-kMNIST-LeNet-' + batch_size + '-' + model_name + '-' + fault_type + '-20-3-async-' + str(i) + '-run'+str(k),'w') as f:
# 							f.write(results.stdout.decode())
# 						print("finished")
# 						print("========================")
# 					else:
# 						print(results.stderr.decode())
# 						print("failed")
# 						print("========================")

# for k in [0,1,2,3]:
# 	for batch_size in batch_sizes:
# 		for i in nums_async:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = ['mpirun', '-n', '21',
# 							'python', 'distributed_nn.py',
# 							'--batch-size=' + batch_size,
# 							'--max-steps', '1000',
# 							'--epochs', '100',
# 							'--network', 'LeNet',
# 							'--mode', model,
# 							'--dataset', 'MNIST',
# 							'--approach', 'baseline',
# 							'--err-mode', fault_type,
# 							'--lr', '0.01',
# 							'--train-dir', 'output/models/async-ft/mnist/20-3/' + fault_name + '/async-' + str(i) + '/run'+str(k)+'/',
# 							'--seed', str(k),
# 							'--accumulative', 'False',
# 							'--worker-fail', '3',
# 							'--fault-thrshld', '3',
# 							'--async-thrshld', str(i),
# 							'--data-distribution', 'different-dist',
# 							'--calculate-cosine', 'False',
# 							'--checkpoint-step', '0',
# 							'--eval-freq', '1',
# 							'--diff-privacy-param', '0']
# 					print("Now running experiments mpirun "+fault_name+" using "+model_name+" using command:")
# 					print(' '.join(args))
# 					results = subprocess.run(args, capture_output=True)
# 					if results.returncode==0 and results.stdout != None:
# 						with open('logs-asyncft-MNIST-LeNet-' + batch_size + '-' + model_name + '-' + fault_type + '-20-3-async-' + str(i) + '-run'+str(k),'w') as f:
# 							f.write(results.stdout.decode())
# 						print("finished")
# 						print("========================")
# 					else:
# 						print(results.stderr.decode())
# 						print("failed")
# 						print("========================")

# fault_type = "normal"
# fault_name = ""
#
# for k in [0,1,2,3]:
# 	for batch_size in batch_sizes:
# 		for model_name, model in zip(model_names, models):
# 			args = ['mpirun', '-n', '21',
# 					'python', 'distributed_nn.py',
# 					'--batch-size=' + batch_size,
# 					'--max-steps', '1000',
# 					'--epochs', '100',
# 					'--network', 'LeNet',
# 					'--mode', 'normal',
# 					'--dataset', 'KMNIST',
# 					'--approach', 'baseline',
# 					'--err-mode', fault_type,
# 					'--lr', '0.01',
# 					'--train-dir', 'output/models/async-ft/kmnist/20-0/run'+str(k)+'/',
# 					'--seed', str(k),
# 					'--accumulative', 'False',
# 					'--worker-fail', '0',
# 					'--fault-thrshld', '0',
# 					'--async-thrshld', '0',
# 					'--data-distribution', 'different-dist',
# 					'--calculate-cosine', 'False',
# 					'--checkpoint-step', '0',
# 					'--eval-freq', '1',
# 					'--diff-privacy-param', '0']
# 			print("Now running experiments mpirun "+fault_name+" using "+model_name+" using command:")
# 			print(' '.join(args))
# 			results = subprocess.run(args, capture_output=True)
# 			if results.returncode==0 and results.stdout != None:
# 				with open('logs-asyncft-kMNIST-LeNet-' + batch_size + '-' + model_name + '-' + fault_type + '-20-0-run'+str(k),'w') as f:
# 					f.write(results.stdout.decode())
# 				print("finished")
# 				print("========================")
# 			else:
# 				print(results.stderr.decode())
# 				print("failed")
# 				print("========================")

# for k in [0,1,2,3]:
# 	for batch_size in batch_sizes:
# 		for i in nums_async:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = ['mpirun', '-n', '21',
# 							'python', 'distributed_nn.py',
# 							'--batch-size=' + batch_size,
# 							'--max-steps', '1000',
# 							'--epochs', '100',
# 							'--network', 'LeNet',
# 							'--mode', 'normal',
# 							'--dataset', 'MNIST',
# 							'--approach', 'baseline',
# 							'--err-mode', fault_type,
# 							'--lr', '0.01',
# 							'--train-dir', 'output/models/async-ft/mnist/20-0/run'+str(k)+'/',
# 							'--seed', str(k),
# 							'--accumulative', 'False',
# 							'--worker-fail', '0',
# 							'--fault-thrshld', '0',
# 							'--async-thrshld', str(i),
# 							'--data-distribution', 'different-dist',
# 							'--calculate-cosine', 'False',
# 							'--checkpoint-step', '0',
# 							'--eval-freq', '1',
# 							'--diff-privacy-param', '0']
# 					print("Now running experiments mpirun "+fault_name+" using "+model_name+" using command:")
# 					print(' '.join(args))
# 					results = subprocess.run(args, capture_output=True)
# 					if results.returncode==0 and results.stdout != None:
# 						with open('logs-asyncft-MNIST-LeNet-' + batch_size + '-' + model_name + '-' + fault_type + '-20-0-run'+str(k),'w') as f:
# 							f.write(results.stdout.decode())
# 						print("finished")
# 						print("========================")
# 					else:
# 						print(results.stderr.decode())
# 						print("failed")
# 						print("========================")


# print()
# model_names = ['async-cge']
# models = ['async_grad_norm']
# fault_types = ['rev_grad_2']#,'normfilter','labelflipping']
# fault_names = ['revgrad2']#,'normfilter','labelflipping']
# nums_async = [0,1,3,5,10]
# batch_sizes = ['128']
# #
# for k in [0,1]:
# 	for batch_size in batch_sizes:
# 		for i in nums_async:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = 'python distributed_eval.py --model-dir output/models/async-ft/kmnist/20-3/'+fault_name+'/async-'+str(i)+'/run'+str(k)\
# 							+'/ --dataset KMNIST --network LeNet --eval-freq 1 > "results-asyncft-kMNIST-LeNet-'+batch_size+'-'+model_name+'-'+fault_type+'-20-3-async-'+str(i)+'-run'+str(k)+'" 2>&1 &'
# 					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
# 					print(args)
# 					results = subprocess.run(args, shell=True)

# for k in [0,1,2,3]:
# 	for batch_size in batch_sizes:
# 		for i in nums_async:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = 'python distributed_eval.py --model-dir output/models/async-ft/mnist/20-3/'+fault_name+'/async-'+str(i)+'/run'+str(k)\
# 							+'/ --dataset MNIST --network LeNet --eval-freq 1 > "results-asyncft-MNIST-LeNet-'+batch_size+'-'+model_name+'-'+fault_type+'-20-3-async-'+str(i)+'-run'+str(k)+'" 2>&1 &'
# 					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
# 					print(args)
# 					results = subprocess.run(args, shell=True)

for k in [0,1]:
	for batch_size in batch_sizes:
		for i in nums_async:
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/async-ft/kmnist/20-0/run'+str(k)\
						+'/ --dataset KMNIST --network LeNet --eval-freq 1 > "results-asyncft-kMNIST-LeNet-'+batch_size+'-'+model_name+'-20-0-run'+str(k)+'" 2>&1 &'
				print("Now evaluating "+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)

# for k in [0,1,2,3]:
# 	for batch_size in batch_sizes:
# 		for i in nums_async:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = 'python distributed_eval.py --model-dir output/models/async-ft/mnist/20-0/run'+str(k)\
# 							+'/ --dataset MNIST --network LeNet --eval-freq 1 > "results-asyncft-MNIST-LeNet-'+batch_size+'-'+model_name+'-'+fault_type+'-20-0-run'+str(k)+'" 2>&1 &'
# 					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
# 					print(args)
# 					results = subprocess.run(args, shell=True)
