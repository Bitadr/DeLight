import numpy as np
from scipy import stats
import cPickle
import os
import math
import sys
import string

np.set_printoptions(threshold='nan')

f = open(sys.argv[1]+".pkl", 'rb')
temp_train, temp_valid, temp_test = cPickle.load(f)
f.close()
tr_data, tr_label = temp_train
tr_label = np.asarray(tr_label)


k = string.atoi(sys.argv[2])
n = tr_data.shape[0]

f = open('H.pkl', 'rb')
H = cPickle.load(f)
f.close()
H = np.asarray(H)

f = open('train_label_c.pkl', 'rb')
pred_label = cPickle.load(f)
f.close()
pred_label = np.asarray(pred_label)


f = open('D.pkl', 'rb')
D = cPickle.load(f)
f.close()
D = np.asarray(D)

temp = 0.
for i in range(n):
		for j in range(k):
			temp += D[i][j]*(1.+H[i][j]-H[i][tr_label[i]])
		
		
temp = temp/2.0
alpha = float(temp)/((1. - temp))


with open(sys.argv[3]+sys.argv[4], "a") as myfile:
	myfile.write(str(alpha))
	myfile.write(" ")

D_new = np.zeros((n, k))	
for i in range(n):
	for j in range(k):
		D_new[i][j] = D[i][j] * math.pow(alpha, ((1./2.)*(1 - H[i][j] + H[i][tr_label[i]])))

D_new = D_new/sum(sum(D_new))
V = np.zeros((n, k))
for i in range(n):
	for j in range(k):
		if (j == tr_label[i]):
			V[i][j] = 1.
		else:
			V[i][j] = D_new[i][j]/np.amax(D_new[i,:])
			
			
save_file = open("D.pkl", 'wb')
cPickle.dump(D_new, save_file, -1) 
save_file.close()

save_file = open("W_V.pkl", 'wb')
cPickle.dump(V, save_file, -1) 
save_file.close()





