import matplotlib.pyplot as plt
import re
import numpy as np
import statistics as stat
import sys, os

#model_names = ['normal', 'krum', 'median_of_means', 'coor_wise_median', 'coor_wise_trimmed_mean', 'grad_norm']
#model_names = ['normal', 'grad_norm-keep_all', 'grad_norm-drop_f']
# model_names = ['normal', 'geomedian', 'medofmeans', 'cwtm', 'mkrum5', 'normfilter']
model_names = ["linreg/baseline/", "linreg/diminlr2/", 'linreg/adapt/', 'linreg/roll/']
# model_names=["test{}/".format(i) for i in range(1,5)]
# for i in range(3,8):
#     for j in range(3,8):
#         model_names.append('p{}_q{}/'.format(i,j))

#model_names = ['normal', 'mkrum5', 'mkrum5mr', 'mkrum52', 'mkrum53', 'mkrum1', 'mkrum2', 'normfilter']
#model_names = ['normal', 'grad_norm-drop-f', 'grad_norm-keep-n-clip', 'grad_norm_full_grad-drop-f', 'grad_norm_full_grad-keep-n-clip', 'grad_norm_multi_parts-20-keep-n-clip']
#model_names = ['normal', 'grad_norm', 'grad_norm_full_grad', 'grad_norm_multi_parts-20']
#model_names = ['normal', 'grad_norm-drop-f', 'grad_norm_full_grad-drop-f']
#model_names = ['normal', 'krum', 'multi_krum-5']
#model_names = ['100','40','25','10','5']

new_ticks = np.linspace(0,100,101)
plt.figure(figsize=(10,5), dpi=1200)
plt.xlim(0,100)
#plt.ylim(-1,1)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Loss")
losses=[]
avg_losses=[]
min_y=100
max_y=-100
max_steps=0
for model_name in model_names:
    step=[]
    loss=[]
    # filename = 'output/models/linreg/' + model_name + 'results.npy'
    filename = 'output/models/' + model_name + 'results.npy'
    print(model_name)
    results = np.load(filename)
    step = results[0].astype(int)
    step = step[1:]
    loss = results[1]
    loss = loss[1:]
    prec1 = results[2]
    prec3 = results[3]
    # plt.plot(step,loss,label=model_name,linewidth=.2, color='grey')
    
    # get the final step of convergence
    ''' temp-- convergence == loss < .005 '''
    conv_step = 0
    while conv_step < len(loss):
        if loss[conv_step] < .005:
            break
        conv_step += 1

    step = step[:conv_step]
    if len(step) > max_steps:
        max_steps = len(step)
    loss = loss[:conv_step]
    
    # get the steps with faults
    faults = []
    if model_name != "cntrl/":
        logfile = 'output/models/' + model_name + 'logfile.txt'
        lf = open(logfile,'r')
        lines = lf.readlines()
        lf.close()
        step_ct = 0
        for line in lines:
            tokens = line.strip().split(' ')
            if (len(tokens) > 12) and (tokens[7] != '0') and (tokens[13]=='1'):
                if step_ct == conv_step:
                    break
                else:
                    step_ct += 1
                faults.append(int(tokens[5])-2)

    plt.plot(step,loss, markevery=faults, marker='o', markersize=1,label=model_name,linewidth=.5)
    losses.append(loss)
    avg_losses.append(loss)
    if np.min(loss) < min_y:
        min_y = np.min(loss)
    if np.max(loss) > max_y:
        max_y = np.max(loss)

# avg_loss = [stat.mean(k) for k in zip(*avg_losses)]
# plt.plot(step,avg_loss,label='mean',linewidth=1.1, color='blue')
# step=[]
# loss=[]
# filename = 'output/models/exp2/control/results.npy'
# results = np.load(filename)
# step = results[0].astype(int)
# step = step[1:]
# loss = results[1]
# loss = loss[1:]
# # plt.plot(step,loss,label='fault-free',linewidth=1.1, color='green')
# losses.append(loss)
# 
# if np.min(loss) < min_y:
#     min_y = np.min(loss)
# if np.max(loss) > max_y:
#     max_y = np.max(loss)

# filename = 'logs-validationloss'
# lines=None
# with open(filename, 'r') as f:
#     lines = f.readlines()
# 
# val_losses=[]
# for line in lines:
#     val_losses.append(float(line.strip().split(' ')[1]))
# val_losses = np.array(val_losses)
# step = np.arange(1.0, 301.0)
# plt.plot(step,val_losses,label='validation', linewidth=.5)

plt.legend(loc="right")
# plt.ylim(0,max_y+.1)
plt.ylim(0,1)
# plt.xlim(0,len(loss))
''' temp '''
plt.xlim(0,max_steps)
# plt.title("n=10, batch_size=48, q=.20, p=.20, f=1, network=FC, err=rev_grad")
# plt.savefig("red_results/exp3/loss-exp2-q2-p2-f1.pdf")
plt.savefig("red_results/linreg_loss5b.pdf")
plt.clf()
""" EXIT PROGRAM (TEMPORARY) """
sys.exit()

new_ticks = np.linspace(0,1000,1001)
plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1000)
#plt.ylim(-1,1)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Prec1")
losses=[]

for model_name in model_names:
    step=[]
    loss=[]
    filename = 'output/models/exp2/' + model_name + 'results.npy'
    #filename = 'results-normal-1-'+model_name+'-0-60000'
    print(model_name)
    results = np.load(filename)
    step = results[0].astype(int)
    loss = results[1]
    prec1 = results[2]
    prec3 = results[3]
    plt.plot(step,prec1,label=model_name,linewidth=.1)
    losses.append(loss)

plt.legend(loc="right")
plt.savefig("red_results/prec1-exp3-500.pdf")
plt.clf()

new_ticks = np.linspace(0,1000,1001)
plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1000)
#plt.ylim(-1,1)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Prec3")
losses=[]

for model_name in model_names:
    step=[]
    loss=[]
    filename = 'output/models/exp2/' + model_name + 'results.npy'
    #filename = 'results-normal-1-'+model_name+'-0-60000'
    print(model_name)
    results = np.load(filename)
    step = results[0].astype(int)
    loss = results[1]
    prec1 = results[2]
    prec3 = results[3]
    plt.plot(step,prec3,label=model_name,linewidth=.1)
    losses.append(loss)

plt.legend(loc="right")
plt.savefig("red_results/prec3-exp3-500.pdf")
plt.clf()

new_ticks = np.linspace(0,1000,1001)
plt.figure(figsize=(10,9), dpi=1200)
plt.xlim(0,1000)
#plt.ylim(-1,1)
#plt.xticks(new_ticks)
plt.xlabel("Step")
plt.ylabel("Comp Eff")
losses=[]

for model_name in model_names:
    step=[]
    loss=[]
    filename = 'output/models/exp2/' + model_name + 'comp_eff.npy'
    #filename = 'results-normal-1-'+model_name+'-0-60000'
    print(model_name)
    results = np.load(filename)
    results = np.delete(results,-1,1)
    step = results[0].astype(int)
    used = results[1]
    total = results[2]
    ce = used/total
    plt.plot(step,ce,label=model_name,linewidth=.6)
    losses.append(loss)

plt.legend(loc="right")
plt.ylim(0,1.05)
plt.savefig("red_results/exp2/compeff-exp2.pdf")

