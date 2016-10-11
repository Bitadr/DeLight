import cPickle, numpy

valid_set_yt = numpy.loadtxt('DAValid_L',delimiter='\t')
valid_set_xt = numpy.loadtxt('DAValid_o',delimiter='\t')

test_set_yt = numpy.loadtxt('DATest_L',delimiter='\t')
test_set_xt = numpy.loadtxt('DATest_o',delimiter='\t')


train_set_yt = numpy.loadtxt('DATrain_L',delimiter='\t')
train_set_xt = numpy.loadtxt('DATrain_o',delimiter='\t')

tup1 = (train_set_xt, train_set_yt)
tup2 = (valid_set_xt, valid_set_yt)
tup3 = (test_set_xt, test_set_yt)

tup = tuple((tup1,tup2,tup3))
save_file = open('DA_o.pkl', 'wb')
cPickle.dump(tup, save_file, -1) 
save_file.close()
