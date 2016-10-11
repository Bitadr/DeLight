import numpy as np
from scipy import stats
import string
import sys
import cPickle
np.set_printoptions(threshold='nan')


f = open(sys.argv[1]+".pkl", 'rb')
temp_train, temp_valid, temp_test = cPickle.load(f)
f.close()
train_set_data, train_set_label = temp_train
valid_set_xt, valid_set_yt = temp_valid
test_set_xt, test_set_yt = temp_test

n = train_set_data.shape[0]
k = string.atoi(sys.argv[2])


f = open("D.pkl", 'rb')
D = cPickle.load(f)
f.close()
D = np.asmatrix(D)

f = open("W_V.pkl", 'rb')
W_V = cPickle.load(f)
f.close()
W_V = np.asmatrix(W_V)

temp = np.zeros((n, 1))
for i in range(n):
	temp[i] = np.sum(D[i,:])
	
p = temp/np.sum(temp)
print p.shape
xk = np.arange(p.shape[0])

custm = stats.rv_discrete(name='custm', values=(xk, p))

ind_train = custm.rvs(size=p.shape[0])


V_shuffled = W_V[ind_train,:]
D_shuffled = D[ind_train, :]
train_set_xt = train_set_data[ind_train, :]
train_set_yt = train_set_label[ind_train]

Z = np.zeros((n, k))
for i in range(n):
	for j in range(k):
		if (j == train_set_yt[i]):
			Z[i][j] = 1.
			

tup1 = (train_set_xt, train_set_yt)
tup2 = (valid_set_xt, valid_set_yt)
tup3 = (test_set_xt, test_set_yt)

tup = tuple((tup1,tup2,tup3))

save_file = open(sys.argv[1]+".pkl", 'wb')
cPickle.dump(tup, save_file, -1) 
save_file.close()

save_file = open("Z.pkl", 'wb')
cPickle.dump(Z, save_file, -1) 
save_file.close()

save_file = open("D.pkl", 'wb')
cPickle.dump(D_shuffled, save_file, -1) 
save_file.close()

save_file = open("W_V.pkl", 'wb')
cPickle.dump(V_shuffled, save_file, -1) 
save_file.close()
