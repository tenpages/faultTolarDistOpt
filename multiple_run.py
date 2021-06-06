import subprocess

model_names = ['cwtm','normfilter']
models = ['coor_wise_trimmed_mean','grad_norm']
fault_types = ['rev_grad_2','labelflipping']
fault_names = ['revgrad2','labelflipping']
nums_faults = [3]
batch_sizes = ['128']
#acc_alpha = '60'
#acc_alphas = ['20','40','60']
total = 10

# for batch_size in batch_sizes:
# 	for i in nums_faults:
# 		for fault_type, fault_name in zip(fault_types, fault_names):
# 			for model_name, model in zip(model_names, models):
# 				args = ['mpirun', '-n', str(total+1),
# 						'python', 'distributed_nn.py',
# 						'--batch-size=' + batch_size,
# 						'--max-steps', '1000',
# 						'--epochs', '100',
# 						'--network', 'LeNet',
# 						'--mode', model,
# 						'--dataset', 'MNIST',
# 						'--approach', 'baseline',
# 						'--err-mode', fault_type,
# 						'--lr', '0.01',
# 						'--train-dir', 'output/models/approx/MNIST/10-'+str(i)+'/'+model_name+'/'+fault_name+'/',
# 						'--seed', '0',
# 						'--accumulative', 'False',
# 						'--worker-fail', str(i),
# 						'--fault-thrshld', str(i),
# 						'--data-distribution', 'same',
# 						'--calculate-cosine', 'False',
# 						'--checkpoint-step', '0',
# 						'--eval-freq', '1',
# 						'--diff-privacy-param', '0']
# 				print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
# 				print(' '.join(args))
# 				results = subprocess.run(args, capture_output=True)
# 				if results.returncode==0 and results.stdout != None:
# 					with open('logs-approx-MNIST-LeNet-' + batch_size + '-' + model_name + '-' + fault_name + '-10-' + str(i),'w') as f:
# 						f.write(results.stdout.decode())
# 					print("finished")
# 					print("========================")
# 				else:
# 					print(results.stderr.decode())
# 					print("failed")
# 					print("========================")
#
# for batch_size in batch_sizes:
# 	for fault_type, fault_name in zip(fault_types, fault_names):
# 		for model_name, model in zip(model_names, models):
# 			args = ['mpirun', '-n', str(total+1),
# 					'python', 'distributed_nn.py',
# 					'--batch-size=' + batch_size,
# 					'--max-steps', '1000',
# 					'--epochs', '100',
# 					'--network', 'LeNet',
# 					'--mode', model,
# 					'--dataset', 'MNIST',
# 					'--approach', 'baseline',
# 					'--err-mode', fault_type,
# 					'--lr', '0.01',
# 					'--train-dir', 'output/models/approx/MNIST/10-3/normal/',
# 					'--seed', '0',
# 					'--accumulative', 'False',
# 					'--worker-fail', '3',
# 					'--fault-thrshld', '3',
# 					'--data-distribution', 'same',
# 					'--calculate-cosine', 'False',
# 					'--checkpoint-step', '0',
# 					'--omit-faults', 'True',
# 					'--eval-freq', '1',
# 					'--diff-privacy-param', '0']
# 			print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
# 			print(' '.join(args))
# 			results = subprocess.run(args, capture_output=True)
# 			if results.returncode==0 and results.stdout != None:
# 				with open('logs-approx-MNIST-LeNet-' + batch_size + '-normal-10-3','w') as f:
# 					f.write(results.stdout.decode())
# 				print("finished")
# 				print("========================")
# 			else:
# 				print(results.stderr.decode())
# 				print("failed")
# 				print("========================")
#
# for batch_size in batch_sizes:
# 	for i in nums_faults:
# 		for fault_type, fault_name in zip(fault_types, fault_names):
# 			for model_name, model in zip(model_names, models):
# 				args = ['mpirun', '-n', str(total+1),
# 						'python', 'distributed_nn.py',
# 						'--batch-size=' + batch_size,
# 						'--max-steps', '1000',
# 						'--epochs', '100',
# 						'--network', 'LeNet',
# 						'--mode', model,
# 						'--dataset', 'Fashion-MNIST',
# 						'--approach', 'baseline',
# 						'--err-mode', fault_type,
# 						'--lr', '0.01',
# 						'--train-dir', 'output/models/approx/Fashion-MNIST/10-'+str(i)+'/'+model_name+'/'+fault_name+'/',
# 						'--seed', '0',
# 						'--accumulative', 'False',
# 						'--worker-fail', str(i),
# 						'--fault-thrshld', str(i),
# 						'--data-distribution', 'same',
# 						'--calculate-cosine', 'False',
# 						'--checkpoint-step', '0',
# 						'--eval-freq', '1',
# 						'--diff-privacy-param', '0']
# 				print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
# 				print(' '.join(args))
# 				results = subprocess.run(args, capture_output=True)
# 				if results.returncode==0 and results.stdout != None:
# 					with open('logs-approx-fMNIST-LeNet-' + batch_size + '-' + model_name + '-' + fault_name + '-10-' + str(i),'w') as f:
# 						f.write(results.stdout.decode())
# 					print("finished")
# 					print("========================")
# 				else:
# 					print(results.stderr.decode())
# 					print("failed")
# 					print("========================")

batch_size = '128'
fault_type = 'rev_grad_2'
fault_name = 'revgrad2'
model_name = 'normal'
args = ['mpirun', '-n', str(total+1),
		'python', 'distributed_nn.py',
		'--batch-size=' + batch_size,
		'--max-steps', '1000',
		'--epochs', '100',
		'--network', 'LeNet',
		'--mode', 'normal',
		'--dataset', 'Fashion-MNIST',
		'--approach', 'baseline',
		'--err-mode', fault_type,
		'--lr', '0.01',
		'--train-dir', 'output/models/approx/Fashion-MNIST/10-3/normal/',
		'--seed', '0',
		'--accumulative', 'False',
		'--worker-fail', '3',
		'--fault-thrshld', '3',
		'--data-distribution', 'same',
		'--calculate-cosine', 'False',
		'--checkpoint-step', '0',
		'--omit-faults', 'True',
		'--eval-freq', '1',
		'--diff-privacy-param', '0']
print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
print(' '.join(args))
results = subprocess.run(args, capture_output=True)
if results.returncode==0 and results.stdout != None:
	with open('logs-approx-fMNIST-LeNet-' + batch_size + '-normal-10-3','w') as f:
		f.write(results.stdout.decode())
	print("finished")
	print("========================")
else:
	print(results.stderr.decode())
	print("failed")
	print("========================")


args = ['mpirun', '-n', str(total+1),
		'python', 'distributed_nn.py',
		'--batch-size=' + batch_size,
		'--max-steps', '1000',
		'--epochs', '100',
		'--network', 'LeNet',
		'--mode', 'normal',
		'--dataset', 'MNIST',
		'--approach', 'baseline',
		'--err-mode', fault_type,
		'--lr', '0.01',
		'--train-dir', 'output/models/approx/MNIST/10-3/normal/',
		'--seed', '0',
		'--accumulative', 'False',
		'--worker-fail', '3',
		'--fault-thrshld', '3',
		'--data-distribution', 'same',
		'--calculate-cosine', 'False',
		'--checkpoint-step', '0',
		'--omit-faults', 'True',
		'--eval-freq', '1',
		'--diff-privacy-param', '0']
print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
print(' '.join(args))
results = subprocess.run(args, capture_output=True)
if results.returncode==0 and results.stdout != None:
	with open('logs-approx-MNIST-LeNet-' + batch_size + '-normal-10-3','w') as f:
		f.write(results.stdout.decode())
	print("finished")
	print("========================")
else:
	print(results.stderr.decode())
	print("failed")
	print("========================")


model_names = ['cwtm','normfilter']
models = ['coor_wise_trimmed_mean','grad_norm']
fault_types = ['rev_grad_2','labelflipping']
fault_names = ['revgrad2','labelflipping']
nums_faults = [3]
batch_sizes = ['128']
#acc_alpha = '60'
#acc_alphas = ['20','40','60']
total = 10

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/approx/Fashion-MNIST/10-'+str(i)+'/'+model_name+'/'+fault_name \
						+'/ --dataset Fashion-MNIST --network LeNet --eval-freq 1 > "results-approx-fMNIST-LeNet-'+batch_size+'-'+model_name+'-'+fault_name+'-10-'+str(k)+'" 2>&1 &'
				print("Now evaluating "+fault_name+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				args = 'python distributed_eval.py --model-dir output/models/approx/MNIST/10-'+str(i)+'/'+model_name+'/'+fault_name \
						+'/ --dataset MNIST --network LeNet --eval-freq 1 > "results-approx-MNIST-LeNet-'+batch_size+'-'+model_name+'-'+fault_name+'-10-'+str(k)+'" 2>&1 &'
				print("Now evaluating "+fault_name+" using "+model_name+" using command:")
				print(args)
				results = subprocess.run(args, shell=True)

batch_size = '128'
fault_name = 'revgrad2'
model_name = 'normal'
args = 'python distributed_eval.py --model-dir output/models/approx/Fashion-MNIST/10-3/normal/' \
		+' --dataset Fashion-MNIST --network LeNet --eval-freq 1 > "results-approx-fMNIST-LeNet-'+batch_size+'-normal-10-3" 2>&1 &'
print("Now evaluating "+fault_name+" using "+model_name+" using command:")
print(args)
results = subprocess.run(args, shell=True)

args = 'python distributed_eval.py --model-dir output/models/approx/MNIST/10-3/normal/' \
		+' --dataset MNIST --network LeNet --eval-freq 1 > "results-approx-MNIST-LeNet-'+batch_size+'-normal-10-3" 2>&1 &'
print("Now evaluating "+fault_name+" using "+model_name+" using command:")
print(args)
results = subprocess.run(args, shell=True)

