import matplotlib.pyplot as plt
import re
import numpy as np

#model_names = ['normal', 'krum', 'median_of_means', 'coor_wise_median', 'coor_wise_trimmed_mean', 'grad_norm']
#model_names = ['normal', 'grad_norm-keep_all', 'grad_norm-drop_f']
model_names = ['normal', 'geomedian', 'medofmeans', 'cwtm', 'mkrum5', 'normfilter']
#model_names = ['normal', 'mkrum5', 'mkrum5mr', 'mkrum52', 'mkrum53', 'mkrum1', 'mkrum2', 'normfilter']
#model_names = ['normal', 'grad_norm-drop-f', 'grad_norm-keep-n-clip', 'grad_norm_full_grad-drop-f', 'grad_norm_full_grad-keep-n-clip', 'grad_norm_multi_parts-20-keep-n-clip']
#model_names = ['normal', 'grad_norm', 'grad_norm_full_grad', 'grad_norm_multi_parts-20']
#model_names = ['normal', 'grad_norm-drop-f', 'grad_norm_full_grad-drop-f']
#model_names = ['normal', 'krum', 'multi_krum-5']
#model_names = ['100','40','25','10','5']
current_step = 0
new_ticks = np.linspace(0,1000,1001)
plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1000)
#plt.ylim(-1,1)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Loss")
losses=[]

for model_name in model_names:
    step=[]
    loss=[]
    if model_name=='normal':
        filename = 'output/models/paper1/MNIST-LeNet/16/' + model_name + '/40-0/results.npy'
    else:
    	filename = 'output/models/paper1/MNIST-LeNet/16/geomedian/' + model_name + '/40-10/results.npy'
    #filename = 'results-normal-1-'+model_name+'-0-60000'
    print(model_name)
    results = np.load(filename)
    step = results[0].astype(int)
    loss = results[1]
    prec1 = resulsts[2]
    prec3 = results[3]
    plt.plot(step,loss,label=model_name,linewidth=.1)
    losses.append(loss)

plt.legend(loc="right")
plt.ylim(0,2.5)
plt.savefig("results/loss-mnist-16-10-geomedian.pdf")
plt.xlim(10000,60000)
plt.savefig("results/loss-normal-1-2.pdf")
plt.clf()

plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1220)
plt.ylim(0,100)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Precision")

for model_name in model_names:
    step=[]
    prec1=[]
    if model_name=='normal':
        filename = 'results-' + model_name + '-41-0-1200'
    else:
        filename = 'results-'+model_name+'-41-10-1200'
    print(model_name)
    with open(filename,"r") as f:
        for line in f:
            t=re.match("Evaluator evaluating results on step (\d+)\D+",line)
            if t!=None:
                current_step = int(t.groups()[0])
            else:
                t=re.match("Test set: Average loss: (.+), Prec@1: (\d+\.\d+) Prec@5: (\d+\.\d+)",line)
                if t!=None:
                    current_prec_1 = float(t.groups()[1])
                    step.append(current_step)
                    prec1.append(current_prec_1)
    plt.plot(step,prec1,label=model_name)

plt.legend(loc="right")
plt.savefig("prec1.pdf")
plt.clf()

plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1220)
plt.ylim(0,100)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Precision")
for model_name in model_names:
    step=[]
    prec5=[]
    filename = 'results-'+model_name+'-41-2-1200'
    print(model_name)
    with open(filename,"r") as f:
        for line in f:
            t=re.match("Evaluator evaluating results on step (\d+)\D+",line)
            if t!=None:
                current_step = int(t.groups()[0])
            else:
                t=re.match("Test set: Average loss: (.+), Prec@1: (\d+\.\d+) Prec@5: (\d+\.\d+)",line)
                if t!=None:
                    current_prec_5 = float(t.groups()[2])
                    step.append(current_step)
                    prec5.append(current_prec_5)
    plt.plot(step,prec5,label=model_name)

plt.legend(loc="right")
plt.savefig("prec5.pdf", dpi=800)
