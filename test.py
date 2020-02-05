from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
status = MPI.Status()

rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
	print("MASTER: initiated")
	receiver=[]
	requests=[]
	for i in range(size-1):
		receiver.append([])
		for j in range(1,6):
			t=np.zeros(j)
			receiver[i].append(t)
			req=comm.Irecv(t, source=i+1, tag=88+j)
			print("MASTER: receiver",t.shape,"in size=",j,"for tag",88+j)
			requests.append(req)
	print("MASTER: receiver constructed")

	count=0
	while count<(size-1)*5:
		MPI.Request.Waitany(requests=requests, status=status)
		print("MASTER: received", type(receiver[status.source-1][status.tag-88-1]), "from", status.source, "for",status.tag)
		count+=1
	#a,b = MPI.Request.waitany(requests=requests)
	#print("received",a,b)
else:
	for i in range(1,6):
		t=np.random.random_sample(i)
		r=comm.Isend(t, dest=0, tag=88+i)
		r.wait()
		print("WORKER #",rank,"sent",t.shape,"with tag",88+i)
	print("WORKER #",rank," transmitted")
