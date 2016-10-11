import numpy as np

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
    def AdaboostCost(self, OurW, OurZ):
    	return (1/2.)*T.sum((OurW*(self.p_y_given_x - OurZ)*(self.p_y_given_x - OurZ)))
    	#return (1./2)*(T.sum((OurW*(self.p_y_given_x - OurZ)*(self.p_y_given_x - OurZ)),axis=1))
    
    def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
           
                 
        
