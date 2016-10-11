import numpy as np
import cPickle
import sys
import string
np.set_printoptions(threshold='nan')

f = open(sys.argv[1]+".pkl", 'rb')
temp_train, temp_valid, temp_test = cPickle.load(f)
f.close()
tr_data, tr_label = temp_train
t_data, t_label = temp_test
v_data, v_label = temp_valid

if (min(tr_label) == 1.):
	tr_label = tr_label - 1.
	t_label = t_label - 1.
	v_label = v_label - 1.
	


np.savetxt('Test_Label',t_label, delimiter = " ", fmt="%f" )

n =  tr_data.shape[0]
k = string.atoi(sys.argv[2])

Z = np.zeros((n, k))
V = np.zeros((n, k))
D = np.zeros((n, k))
for i in range(n):
	for j in range(k):
		V[i][j] = 1.
		if (j == tr_label[i]):
			Z[i][j] = 1.

			
	
for i in range(n):
	for j in range(k):
		if (j == tr_label[i]):
			D[i, j] = 0.
		else:
			D[i, j] = 1./(n*(k-1.))
			

save_file = open('W_V.pkl', 'wb')
cPickle.dump(V, save_file, -1) 
save_file.close()


save_file = open('Z.pkl', 'wb')
cPickle.dump(Z, save_file, -1) 
save_file.close()


save_file = open('D.pkl', 'wb')
cPickle.dump(D, save_file, -1) 
save_file.close()


tup1 = (tr_data, tr_label)
tup2 = (t_data, t_label)
tup3 = (v_data, v_label)

tup = tuple((tup1,tup2,tup3))

save_file = open(sys.argv[1]+".pkl", 'wb')
cPickle.dump(tup, save_file, -1) 
save_file.close()






