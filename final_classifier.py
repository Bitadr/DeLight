import numpy as np
import cPickle
import os
import math
import array
import string
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold='nan')

N_c = string.atoi(sys.argv[1]) 
M = map(int, sys.argv[2].split(',')) 
f = open(sys.argv[3]+".pkl", 'rb')
temp_train, temp_valid, temp_test = cPickle.load(f)
f.close()
test_set_xt, test_true_label = temp_test
valid_set_xt, valid_true_label = temp_valid

	

N_classifier = map(int, sys.argv[4].split(','))

for f in range(len(N_classifier)):

	text_file = open("Alpha"+str(N_classifier[f]+1), "r")
	alpha = text_file.read().split(' ')
	alpha = map(float,alpha[:-1])
	text_file.close()
	
	findc = [0]* M[f]
	
	for k in range(M[f]):
	
		f1 = open("valid_ff_H"+str(k+1)+str(N_classifier[f]+1)+".pkl",'rb')
		validff_h = cPickle.load(f1)
		f1.close()
		
		f2 = open("test_ff_H"+str(k+1)+str(N_classifier[f]+1)+".pkl",'rb')
		testff_h = cPickle.load(f2)
		f2.close()
		
		if k==0:
			cc = np.zeros((testff_h.shape[0], N_c))
			ch = np.zeros((validff_h.shape[0], N_c))
		
		testff_h = np.asarray(testff_h)
		validff_h = np.asarray(validff_h)
	
		for i in range (testff_h.shape[0]):
			cc[i, :] += (np.log(1./alpha[k]))*testff_h[i,:]
		for i in range (validff_h.shape[0]):
			ch[i, :] += (np.log(1./alpha[k]))*validff_h[i,:]
		

	ada_label_valid = np.argmax(ch, axis=1)
	err_rate_valid = 0.
	
	for i in range (valid_true_label.shape[0]):
		if ada_label_valid[i]!=valid_true_label[i]:
			err_rate_valid+=1.

	acc_rate_valid = 1-((err_rate_valid/float(valid_true_label.shape[0])))

	
	if f==0:
		cmain = np.zeros((testff_h.shape[0], N_c))
		ada_label_temp = np.zeros((testff_h.shape[0], N_c))
	
	for i in range (testff_h.shape[0]):
			##ada_label_temp[i, np.argmax(cc[i,:])] += acc_rate_valid 
			#cmain[i, :] += (np.log(1./alpha_class))*cc[i, :]
			#cmain[i, :] += acc_rate_valid*cc[i, :]
			cmain[i, :] += cc[i, :]
	
##ada_label = np.argmax(ada_label_temp, axis=1)				 
ada_label = np.argmax(cmain, axis=1)
err_rate = 0.
for i in range (test_true_label.shape[0]):
	if ada_label[i]!=test_true_label[i]:
		err_rate+=1.

err_rate = (err_rate/float(test_true_label.shape[0]))*100.
		
print err_rate

