import subprocess

#model_names = ['geomedian','medofmeans','cwtm','mkrum5','normfilter']
#models = ['geometric_median','median_of_means','coor_wise_trimmed_mean','multi_krum','grad_norm']
#fault_types = ['revgrad2','labelflipping','gaussian']
#model_names = ['bulyannormfilter']
#models = ['bulyan_grad_norm']
#model_names = ['mkrum5','cwtm',"medofmeans"]
#models = ['multi_krum','coor_wise_trimmed_mean','median_of_means']
model_names = ['mkrum2', 'normfilter']
models = ['multi_krum', 'grad_norm']
fault_types = ['rev_grad_2']#,'normfilter','labelflipping']
fault_names = ['revgrad2']#,'normfilter','labelflipping']
nums_faults = [3]
batch_sizes = ['32']
df_betas = ['1', '2']
df_sigma = 0.2
#acc_alphas = ['20','40','60']

# for df_beta in df_betas:
# 	for batch_size in batch_sizes:
# 		for i in nums_faults:
# 			for fault_type, fault_name in zip(fault_types, fault_names):
# 				for model_name, model in zip(model_names, models):
# 					args = ['mpirun', '-n', '10',
# 							'python', 'distributed_nn.py',
# 							'--batch-size=' + batch_size,
# 							'--max-steps', '1000',
# 							'--epochs', '100',
# 							'--network', 'LeNet',
# 							'--mode', model, '--multi-krum-m','2',
# 							'--dataset', 'MNIST',
# 							'--approach', 'baseline',
# 							'--err-mode', fault_type,
# 							'--lr', '0.01',
# 							'--train-dir', 'output/models/df/df' + df_beta + '/' + fault_name + '/' + model_name + '/10-' + str(i) + '/',
# 							'--accumulative', 'False',
# 							#'--accumulative-alpha', '0.'+acc_alpha,
# 							'--worker-fail', str(i),
# 							'--fault-thrshld', str(i),
# 							'--data-distribution', 'same',
# 							'--calculate-cosine', 'False',
# 							'--checkpoint-step', '0',
# 							'--eval-freq', '1',
# 							'--diff-privacy-param', df_beta,
# 							'--diff-privacy-sigma', str(df_sigma)]
# 					print("Now running experiments mpiron "+fault_name+" using "+model_name+" using command:")
# 					print(' '.join(args))
# 					results = subprocess.run(args, capture_output=True)
# 					if results.returncode==0 and results.stdout != None:
# 						with open('logs-df-MNIST-LeNet-df(' + df_beta + ',' + str(df_sigma) + ')' + batch_size + '-' + fault_name + '-' + model_name + '-10-' + str(i),'w') as f:
# 							f.write(results.stdout.decode())
# 						print("finished")
# 						print("========================")
# 					else:
# 						print(results.stderr.decode())
# 						print("failed")
# 						print("========================")

print()
for df_beta in df_betas:
	for batch_size in batch_sizes:
		for i in nums_faults:
			for fault_type, fault_name in zip(fault_types, fault_names):
				for model_name, model in zip(model_names, models):
					args = 'python distributed_eval.py --model-dir output/models/df/df'+df_beta+'/'+fault_name+'/'+model_name \
						+'/10-'+str(i)+'/ --dataset MNIST --network LeNet --eval-freq 1 > results-df-MNIST-LeNet-df(' + df_beta + ',' + str(df_sigma) + ')'+batch_size+'-'+fault_name+'-'+model_name+'-10-'+str(i)+' 2>&1 &'
					print("Now evaluating "+fault_name+" using "+model_name+" using command:")
					print(args)
					results = subprocess.run(args, shell=True)
