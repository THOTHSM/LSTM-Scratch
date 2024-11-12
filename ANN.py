import numpy as np
class DenseLayer:
    def __init__(self,input_size,number_of_neurons,l1_weight_regulizer=0,l1_bias_regulizer=0,
                 l2_weight_regulizer=0,l2_bias_regulizer=0):
        self.weights = 0.01*np.random.randn(input_size,number_of_neurons)
        self.bias = np.zeros((1,number_of_neurons))
        self.l1_weight_regulizer = l1_weight_regulizer
        self.l1_bias_regulizer = l1_bias_regulizer
        self.l2_weight_regulizer = l2_weight_regulizer
        self.l2_bias_regulizer = l2_bias_regulizer

    def forward(self,inputs):
        self.inputs = inputs 
        self.output = np.dot(self.inputs,self.weights)+self.bias

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbias = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

        if self.l1_weight_regulizer>0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights<0] = -1
            self.dweights += self.l1_weight_regulizer * dl1

        if self.l1_bias_regulizer>0:
            dl1 = np.ones_like(self.bias)
            dl1[self.bias<0] = -1
            self.dbias += self.l1_bias_regulizer * dl1

        if self.l2_weight_regulizer >0:
            self.dweights += 2*self.weights*self.l2_weight_regulizer

        if self.l2_bias_regulizer >0:
            self.dbias += 2*self.bias*self.l2_bias_regulizer

class ReluActivation:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0]=0
        
        
class Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        
class Loss:
    def regularization_loss(self,layer):
        regularization_loss = 0
        if layer.l1_weight_regulizer > 0:
            regularization_loss += layer.l1_weight_regulizer * np.sum(np.abs(layer.weights))
        if layer.l1_bias_regulizer > 0:
            regularization_loss += layer.l1_bias_regulizer * np.sum(np.abs(layer.bias))
        if layer.l2_weight_regulizer > 0:
            regularization_loss += layer.l2_weight_regulizer * np.sum(layer.weights * layer.weights)
        if layer.l2_bias_regulizer > 0:
            regularization_loss += layer.l2_bias_regulizer * np.sum(layer.bias * layer.bias)
        
        return regularization_loss
    
    def calculate_loss(self,y_pred,y_actual):
        log_likelihood = self.forward(y_pred,y_actual)
        loss = np.mean(log_likelihood)
        return loss 

class CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_actual):
        y_pred_clip = np.clip(y_pred,1e-7,1-1e-7)

        if (y_actual.ndim == 1):
            correct_confidence  = y_pred_clip[range(len(y_pred)),y_actual]
        elif (y_actual.ndim==2):
            correct_confidence = np.sum(y_pred_clip*y_actual,axis=1)
        else:
            raise ValueError("y_actual should be 1-dimensional (label encoded) or 2-dimensional (one-hot encoded).")

        log_likelihood = -np.log(correct_confidence)
        return log_likelihood

class Softmax_CategoricalCrossEntropy:
    def __init__(self):
        self.softmax = Softmax()
        self.CategoricalCrossEntropy = CategoricalCrossEntropy()
    
    def forward(self,inputs,y_actual):
        self.softmax.forward(inputs)
        self.output = self.softmax.output
        return self.CategoricalCrossEntropy.calculate_loss(self.output,y_actual)

    def backward(self,dvalues,y_actual):
        if (y_actual.ndim == 1):
            num_of_classes = np.max(y_actual)+1
            y_actual = np.eye(num_of_classes)[y_actual]
        self.dinputs = (dvalues - y_actual) / len(dvalues)

class Dropouts:
    def __init__(self, dropout_rate):
        self.rate = 1 - dropout_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binomial_matrix = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binomial_matrix

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binomial_matrix

class Metric:
    def accuracy(self, y_pred, y_actual):
        if y_actual.ndim == 2:
            y_actual = np.argmax(y_actual, axis=1)

        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.round(np.mean(y_pred == y_actual), 4)
        return accuracy
    
class SGD_Optimizer:
    def __init__(self,learning_rate=1,decay=0,momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate=learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
    
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = (self.learning_rate)*(1.0/(1.+(self.decay*self.iteration)))

    def parameter_update(self,layer):
        if self.momentum:
            if not hasattr(layer,'momentum_weights'):
                layer.momentum_weights = np.zeros_like(layer.weights)
                layer.momentum_bias = np.zeros_like(layer.bias)
            
            weight_updates = self.momentum * layer.momentum_weights - \
                             self.current_learning_rate * layer.dweights
            layer.momentum_weights = weight_updates

            bias_updates = self.momentum * layer.momentum_bias - \
                           self.current_learning_rate * layer.dbias
            layer.momentum_bias = bias_updates
            
    
        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbias
            
        layer.weights += weight_updates
        layer.bias += bias_updates
        
    def post_update(self):
            self.iteration += 1

class Adagrad_Optimizer:
    def __init__(self,learning_rate=1,decay=0,epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iteration=0

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = (self.learning_rate)*(1./(1.+self.decay*self.iteration))
    
    def parameter_update(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
        
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbias**2

        layer.weights += (-self.current_learning_rate*layer.dweights)/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.bias += (-self.current_learning_rate*layer.dbias)/(np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update(self):
        self.iteration += 1

class RMSProp_Optimizer:
    def __init__(self,learning_rate=0.02,decay=1e-5,epsilon=1e-7,rho=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho=rho
        self.iteration=0

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+self.decay*self.iteration))
    
    def parameter_update(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
        
        layer.weight_cache = layer.weight_cache*self.rho+(1-self.rho)*layer.dweights**2
        layer.bias_cache = layer.bias_cache*self.rho+(1-self.rho)*layer.dbias**2

        layer.weights += -self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.bias += -self.current_learning_rate*layer.dbias/(np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update(self):
        self.iteration += 1

class Adam_Optimizer:
    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.iteration=0

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+self.decay*self.iteration))
    
    def parameter_update(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
            layer.momentum_weights = np.zeros_like(layer.weights)
            layer.momentum_bias = np.zeros_like(layer.bias)
            
        layer.weight_cache = layer.weight_cache*self.beta_2+(1-self.beta_2)*layer.dweights**2
        layer.bias_cache = layer.bias_cache*self.beta_2+(1-self.beta_2)*layer.dbias**2

        corrected_weight_cache = layer.weight_cache/(1-self.beta_2**(self.iteration+1))
        corrected_bias_cache = layer.bias_cache/(1-self.beta_2**(self.iteration+1))

        layer.momentum_weights = layer.momentum_weights*self.beta_1 + (1-self.beta_1)*layer.dweights
        layer.momentum_bias = layer.momentum_bias*self.beta_1 + (1-self.beta_1)*layer.dbias

        corrected_momentum_weights = layer.momentum_weights/(1-self.beta_1**(self.iteration+1))
        corrected_momentum_bias = layer.momentum_bias/(1-self.beta_1**(self.iteration+1))

        layer.weights += -self.current_learning_rate * corrected_momentum_weights /(np.sqrt(corrected_weight_cache)+self.epsilon)
        layer.bias += -self.current_learning_rate * corrected_momentum_bias /(np.sqrt(corrected_bias_cache)+self.epsilon)

    def post_update(self):
        self.iteration += 1 