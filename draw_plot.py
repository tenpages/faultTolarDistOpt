import matplotlib.pyplot as plt
import re
import numpy as np

model_names = ['normal', 'krum', 'median_of_means', 'coor_wise_median', 'coor_wise_trimmed_mean', 'grad_norm']
current_step = 0
new_ticks = np.linspace(0,1220,62)
plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1220)
plt.ylim(0,100)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Precision")

for model_name in model_names:
    step=[]
    prec1=[]
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
