import subprocess

#model_names = ['geomedian','medofmeans','cwtm','mkrum5','normfilter']
#models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum','grad_norm']
#fault_types = ['revgrad2','labelflipping','gaussian']
#model_names = ['bulyannormfilter']
#models = ['bulyan_grad_norm']
model_names = ['mkrum5','cwtm',"medofmeans"]
models = ['multi_krum','coor_wise_trimmed_mean','median_of_means']
fault_types = ['normfilter']
fault_names = ['normfilter']
nums_faults = [8,10]
batch_sizes = ['64']
acc_alphas = ['20','40','60']

for batch_size in batch_sizes:
	for i in nums_faults:
		for fault_type, fault_name in zip(fault_types, fault_names):
			for model_name, model in zip(model_names, models):
				for acc_alpha in acc_alphas:
					args = ['mpirun', '-n', '41',
							'python', 'distributed_nn.py',
							'--batch-size=' + batch_size,
							'--max-steps', '1200',
							'--epochs', '100',
							'--network', 'LeNet',
							'--mode', model, '--multi-krum-m','5',
							'--dataset', 'MNSIT',
							'--approach', 'baseline',
							'--err-mode', fault_type,
							'--lr', '0.01',
							'--train-dir', 'output/models/paper2/MNSIT-LeNet/' + batch_size + '/' + fault_type + '/' + model_name + '/acc-' + acc_alpha + '/40-' + str(i) + '/',
							'--accumulative', 'True',
							'--accumulative-alpha', '0.'+acc_alpha,
							'--worker-fail', str(i),
							'--fault-thrshld', str(i),
							'--data-distribution', 'same',
							'--calculate-cosine', 'False',
							'--checkpoint-step', '500',
							'--eval-freq', '1']
					print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
					print(' '.join(args))
					results = subprocess.run(args, capture_output=True)
					if results.returncode==0 and results.stdout != None:
						with open('logs-paper2-mnist-' + batch_size + '-' + fault_name + '-' + model_name + '-40-' + str(i) + '-acc-' + acc_alpha,'w') as f:
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
				for acc_alpha in acc_alphas:
					args = 'python distributed_eval.py --model-dir output/models/paper2/MNSIT-LeNet/'+batch_size+'/'+fault_name+'/'+model_name \
						+'/acc-'+acc_alpha+'/40-'+str(i)+'/ --dataset MNSIT --network LeNet --eval-freq 1 > results-paper2-mnist-'+batch_size+'-'+fault_name+'-'+model_name+'-40-'+str(i)+'-acc-'+acc_alpha+' 2>&1 &'
					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
					print(args)
					results = subprocess.run(args, shell=True)
