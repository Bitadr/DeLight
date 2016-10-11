import numpy as np
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict
import subprocess
from optparse import OptionParser
import string
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams
 
from logistic_sgd2 import LogisticRegression
from logistic_sgd import load_data

np.set_printoptions(threshold='nan')

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
    
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
    
def Tanh(x):
    y = T.tanh(x)
    return(y)
     
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):
 
        self.input = input
        self.activation = activation
 
        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
         
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
 
        self.W = W
        self.b = b
 
        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
 
def _dropout_from_layer(rng, layer, p):
	srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
	mask = srng.binomial(n=1, p=1-p, size=layer.shape)
	output = layer * T.cast(mask, theano.config.floatX)
	return output
 
class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

 
class MLP(object):
	def __init__(self, 
			rng, 
			input, 
			layer_sizes, 
			dropout_rates, 
			activations, 
			adaweights,
			adaouts,
			use_bias=True):
			
		weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
		self.OurW = adaweights
		self.OurZ = adaouts
		self.layers = []
		self.dropout_layers = []
		next_layer_input = input
		next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
		layer_counter = 0        
		for n_in, n_out in weight_matrix_sizes[:-1]:
			next_dropout_layer = DropoutHiddenLayer(rng=rng, 
				input=next_dropout_layer_input, 
				activation=activations[layer_counter], 
				n_in=n_in, 
				n_out=n_out, 
				use_bias=use_bias, 
				dropout_rate=dropout_rates[layer_counter + 1])
			self.dropout_layers.append(next_dropout_layer)
			next_dropout_layer_input = next_dropout_layer.output
			next_layer = HiddenLayer(rng=rng, 
				input=next_layer_input, 
				activation=activations[layer_counter], 
				W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
				b=next_dropout_layer.b, 
				n_in=n_in, 
				n_out=n_out, 
				use_bias=use_bias)
			self.layers.append(next_layer)
			next_layer_input = next_layer.output
			layer_counter += 1
		
		n_in, n_out = weight_matrix_sizes[-1]
		dropout_output_layer = LogisticRegression(input=next_dropout_layer_input, n_in=n_in, n_out=n_out)
		self.dropout_layers.append(dropout_output_layer)
		output_layer = LogisticRegression(input=next_layer_input, 
			W=dropout_output_layer.W * (1 - dropout_rates[-1]), 
			b=dropout_output_layer.b, 
			n_in=n_in, 
			n_out=n_out)
		self.layers.append(output_layer)
		self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
		self.dropout_errors = self.dropout_layers[-1].errors
		self.AdaboostCost =  self.dropout_layers[-1].AdaboostCost
		self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
		self.errors = self.layers[-1].errors
		self.params = [ param for layer in self.dropout_layers for param in layer.params ]
 
 
def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        dataset,
        W_V,
        Z,
        use_bias,
        random_seed=1234):

    assert len(layer_sizes) - 1 == len(dropout_rates)
     
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
     
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    W_V = theano.shared(np.asmatrix(W_V,dtype=theano.config.floatX),name='W_V')
    Z = theano.shared(np.asmatrix(Z,dtype=theano.config.floatX),name='Z')
 
    print '... building the model'
 
    index = T.lscalar()    
    epoch = T.scalar()
    x = T.matrix('x') 
    y = T.ivector('y')  
    
    OurW = T.matrix('OurW')
    OurZ = T.matrix('OurZ')
    
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))
 
    rng = np.random.RandomState(random_seed)
 
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     adaweights = OurW,
                     adaouts = OurZ,
                     use_bias=use_bias)
 
 
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
                
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
     
    cost = classifier.AdaboostCost(OurW, OurZ)
    dropout_cost = classifier.AdaboostCost(OurW, OurZ)
      
        
    gparams = []
    for param in classifier.params:
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)
 
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)
 
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)
 
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):       
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam
 
    for param, gparam_mom in zip(classifier.params, gparams_mom):
        stepped_param = param + updates[gparam_mom]
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
 	output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={x: train_set_x[index * batch_size:(index + 1) * batch_size],
                OurW: W_V[index * batch_size:(index + 1) * batch_size],
                OurZ: Z[index * batch_size:(index + 1) * batch_size]})

    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})
 
  
    print '... training'
    
    best_params = None
    best_validation_errors = np.inf
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock() 
    done_looping = False
    patience = 5000  
    patience_increase = 2  
    improvement_threshold = 0.995  
    f = open("temp.pkl", 'rb')
    temp_train, temp_valid, temp_test = cPickle.load(f)
    f.close()
    test_set_xt, test_true_label = temp_test

    validation_frequency = min(n_train_batches, patience / 2)

 
    while epoch_counter < n_epochs and (not done_looping):
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index)
            iter = (epoch_counter - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
        		validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        		this_validation_errors = float(np.sum(validation_losses)) / valid_set_x.get_value(borrow=True).shape[0]
        
        		if this_validation_errors < best_validation_errors:
        			if (this_validation_errors < best_validation_errors * improvement_threshold):
        				patience = max(patience, iter * patience_increase)
					best_validation_errors = this_validation_errors
					best_iter = iter
					best_params = param
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = float(np.sum(test_losses)) / test_set_x.get_value(borrow=True).shape[0]
                		
            if patience <= iter:
            	done_looping = True
            	break 
        if epoch_counter % 20 == 0:
        
        	test_model_final = theano.function(inputs=[index],
            	outputs=classifier.layers[-1].y_pred,
            	givens={
            	    x: test_set_x,
                	y: test_set_y},
            	on_unused_input='ignore')
               	 
        	counter = 0.
    		for i in xrange(test_model_final(1).shape[0]):
    			if (test_model_final(1)[i] == test_true_label[i]):
    				counter += 1.
    		print ((test_model_final(1).shape[0] - counter)/test_model_final(1).shape[0])*100.
    		
    	new_learning_rate = decay_learning_rate()
    	 
	end_time = time.clock() 
 	
	
 

    train_model_final = theano.function(inputs=[index],
            outputs=classifier.layers[-1].p_y_given_x,
            givens={
                x: train_set_x,
                y: train_set_y},
            on_unused_input='ignore')
            
    save_file = open('H.pkl', 'wb')
    cPickle.dump(train_model_final(1), save_file, -1) 
    save_file.close()
    
    train_model_final2 = theano.function(inputs=[index],
            outputs=classifier.layers[-1].y_pred,
            givens={
                x: train_set_x,
                y: train_set_y},
            on_unused_input='ignore')
    save_file = open('train_label_c.pkl', 'wb')
    cPickle.dump(train_model_final2(1), save_file, -1) 
    save_file.close()

    
    test_model_final = theano.function(inputs=[index],
            outputs=classifier.layers[-1].p_y_given_x,
            givens={
                x: test_set_x,
                y: test_set_y},
            on_unused_input='ignore')
    save_file = open("test_ff_H"+testindex+classifierindex+".pkl", 'wb')
    cPickle.dump(test_model_final(1), save_file, -1) 
    save_file.close()
    
    test_model_final2 = theano.function(inputs=[index],
            outputs=classifier.layers[-1].p_y_given_x,
            givens={
                x: valid_set_x,
                y: valid_set_y},
            on_unused_input='ignore')
    save_file = open("valid_ff_H"+testindex+classifierindex+".pkl", 'wb')
    cPickle.dump(test_model_final2(1), save_file, -1) 
    save_file.close()
    
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))  
    


 
if __name__ == '__main__':
    import sys
    
    random_seed = 1234
    initial_learning_rate = 1.0
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    
    if len(sys.argv) < 8:
        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        exit(1)
 
    elif sys.argv[1] == "dropout":
        dropout = True
        results_file_name = "rdropout.txt"
 
    elif sys.argv[1] == "backprop":
        dropout = False
        results_file_name = "rbackprop.txt"
 
    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)
        
    layer_sizes = map(int ,sys.argv[2].split(','))
    batch_size = string.atoi(sys.argv[3]) 
    dropout_rates = map(float ,sys.argv[4].split(','))
    activations = map(int, sys.argv[5].split(','))
    n_epochs = string.atoi(sys.argv[6]) 
    dataset = sys.argv[7]
    testindex = sys.argv[8]
    classifierindex = sys.argv[9]
   
    
    f = open('W_V.pkl', 'rb')
    W_V = cPickle.load(f)
    f.close()
    W_V = np.asmatrix(W_V)
    
    f = open('Z.pkl', 'rb')
    Z = cPickle.load(f)
    f.close()
    Z = np.asmatrix(Z)
    
    
    for i in range(len(activations)):
    	if activations[i]==0:
    		activations[i] = Tanh
    	elif activations[i]==1:
    		activations[i] = ReLU
    	else:
    		activations[i] = Sigmoid
    	
    mom_start = 0.5
    mom_end = 0.99
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
                   
 	
    test_mlp(initial_learning_rate=initial_learning_rate,
             learning_rate_decay=learning_rate_decay,
             squared_filter_length_limit=squared_filter_length_limit,
             n_epochs=n_epochs,
             batch_size=batch_size,
             layer_sizes=layer_sizes,
             mom_params=mom_params,
             activations=activations,
             dropout=dropout,
             dropout_rates=dropout_rates,
             dataset=dataset,
             results_file_name=results_file_name,
             W_V = W_V,
             Z = Z,
             use_bias=False,
             random_seed=random_seed)


