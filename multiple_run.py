import subprocess

model_names = ['geomedian','medofmeans','cwtm','mkrum5']
models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum']
#fault_types = ['revgrad2','labelflipping','gaussian']
#model_names = ['bulyannormfilter']
#models = ['bulyan_grad_norm']
#model_names = ['normfilter']
#models = ['grad_norm']
fault_types = ['normfilter','labelflipping']
fault_names = ['normfilter','labelflipping']
nums_faults = [8]
batch_sizes = ['64']
#acc_alphas = ['20','40','60']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				for acc_alpha in acc_alphas:
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
							'--train-dir', 'output/models/paper2/CIFAR10-LeNet/' + batch_size + '/' + fault_name + '/' + model_name + '/40-' + str(i) + '/',
							'--accumulative', 'False',
							#'--accumulative-alpha', '0.'+acc_alpha,
							'--worker-fail', str(i),
							'--fault-thrshld', str(i),
							'--data-distribution', 'same',
							'--calculate-cosine', 'False',
							#'--checkpoint-step', '500',
							'--eval-freq', '1']
					print("Now running experiments on "+fault_name+" using "+model_name+" using command:")
					print(' '.join(args))
					results = subprocess.run(args, capture_output=True)
					if results.returncode==0 and results.stdout != None:
						with open('logs-paper2-cifar-' + batch_size + '-' + fault_name + '-' + model_name + '-40-' + str(i) + '-acc-' + acc_alpha,'w') as f:
							f.write(results.stdout.decode())
						print("finished")
						print("========================")
					else:
						print(results.stderr.decode())
						print("failed")
						print("========================")

model_names = ['geomedian','medofmeans','cwtm','mkrum5','normfilter']
models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum','grad_norm']
fault_types = ['rev_grad_2']
fault_names = ['revgrad2']
nums_faults = [4,8]
batch_sizes = ['64']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				for acc_alpha in acc_alphas:
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
							'--train-dir', 'output/models/paper2/CIFAR10-LeNet/' + batch_size + '/' + fault_name + '/' + model_name + '/40-' + str(i) + '/',
							'--accumulative', 'False',
							#'--accumulative-alpha', '0.'+acc_alpha,
							'--worker-fail', str(i),
							'--fault-thrshld', str(i),
							'--data-distribution', 'same',
							'--calculate-cosine', 'False',
							#'--checkpoint-step', '500',
							'--eval-freq', '1']
					print("Now running experiments on "+fault_name+" using "+model_name+" using command:")
					print(' '.join(args))
					results = subprocess.run(args, capture_output=True)
					if results.returncode==0 and results.stdout != None:
						with open('logs-paper2-cifar-' + batch_size + '-' + fault_name + '-' + model_name + '-40-' + str(i) + '-acc-' + acc_alpha,'w') as f:
							f.write(results.stdout.decode())
						print("finished")
						print("========================")
					else:
						print(results.stderr.decode())
						print("failed")
						print("========================")

print()
model_names = ['geomedian','medofmeans','cwtm','mkrum5']
models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum']
fault_types = ['normfilter','labelflipping']
fault_names = ['normfilter','labelflipping']
nums_faults = [8]
batch_sizes = ['64']
for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				for acc_alpha in acc_alphas:
					args = 'python distributed_eval.py --model-dir output/models/paper2/MNIST-LeNet/'+batch_size+'/'+fault_name+'/'+model_name \
						+'/40-'+str(i)+'/ --dataset CIFAR10 --network LeNet --eval-freq 1 > results-paper2-mnist-'+batch_size+'-'+fault_name+'-'+model_name+'-40-'+str(i)+' 2>&1 &'
					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
					print(args)
					results = subprocess.run(args, shell=True)

model_names = ['geomedian','medofmeans','cwtm','mkrum5','normfilter']
models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum','grad_norm']
fault_types = ['rev_grad_2']
fault_names = ['revgrad2']
nums_faults = [4,8]
for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				for acc_alpha in acc_alphas:
					args = 'python distributed_eval.py --model-dir output/models/paper2/MNIST-LeNet/'+batch_size+'/'+fault_name+'/'+model_name \
						+'/40-'+str(i)+'/ --dataset CIFAR10 --network LeNet --eval-freq 1 > results-paper2-mnist-'+batch_size+'-'+fault_name+'-'+model_name+'-40-'+str(i)+' 2>&1 &'
					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
					print(args)
					results = subprocess.run(args, shell=True)
