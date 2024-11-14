import numpy as np

class Tanh:
    def forward(self,inputs):
        self.outputs = np.tanh(inputs)

    def backward(self,dvalues):
        derivative_of_tanh = 1-self.outputs**2
        self.dinputs = np.multiply(derivative_of_tanh,dvalues)


class Sigmoid:
    def forward(self,inputs):
        inputs = np.clip(inputs, -500, 500)
        sigmoid = 1/(1+np.exp(-inputs))
        self.outputs = sigmoid

    def backward(self,dvalues):
        sigmoid = self.outputs
        derivative_of_sigmoid = np.multiply(sigmoid,(1-sigmoid))
        self.dinputs = np.multiply(derivative_of_sigmoid,dvalues)
                

class LSTM:
    def __init__(self,no_neurons):
        self.no_neurons = no_neurons

        #Weights for forget gates
        self.Uf = 0.1*np.random.randn(self.no_neurons,1)
        self.Wf = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.bf = 0.1*np.random.randn(self.no_neurons,1)

        #Weights for the inputgate
        self.Ui = 0.1*np.random.randn(self.no_neurons,1)
        self.Wi = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.bi = 0.1*np.random.randn(self.no_neurons,1)

        #weights for output gate
        self.Uo = 0.1*np.random.randn(self.no_neurons,1)
        self.Wo = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.bo = 0.1*np.random.randn(self.no_neurons,1)

        #Weights for c-tilda 
        self.Ug = 0.1*np.random.randn(self.no_neurons,1)
        self.Wg = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.bg = 0.1*np.random.randn(self.no_neurons,1)

    def forward(self,x_data):
        self.x = x_data
        self.T = max(x_data.shape)

        self.H = [np.zeros((self.no_neurons,1)) for _ in range(self.T+1)] # Long tern neurons stored acrros all time
        self.C = [np.zeros((self.no_neurons,1)) for _ in range(self.T+1)] # Short term neuron stored across all time
        self.C_tilda = [np.zeros((self.no_neurons,1)) for _ in range(self.T)] # Connection to c after tanh for all time
        self.F = [np.zeros((self.no_neurons,1)) for _ in range(self.T)] # Forget gate neurons for all time 
        self.I = [np.zeros((self.no_neurons,1)) for _ in range(self.T)] # Inputgate neurons for all time
        self.O = [np.zeros((self.no_neurons,1)) for _ in range(self.T)] # output gate neurons (just before combining ht and ct) for all time

        # derivative of Weights for forget gates
        self.dUf = 0.1*np.random.randn(self.no_neurons,1)
        self.dWf = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.dbf = 0.1*np.random.randn(self.no_neurons,1)

        # derivative of Weights for the inputgate
        self.dUi = 0.1*np.random.randn(self.no_neurons,1)
        self.dWi = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.dbi = 0.1*np.random.randn(self.no_neurons,1)

        # derivative of weights for output gate
        self.dUo = 0.1*np.random.randn(self.no_neurons,1)
        self.dWo = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.dbo = 0.1*np.random.randn(self.no_neurons,1)

        # derivative of Weights for c-tilda 
        self.dUg = 0.1*np.random.randn(self.no_neurons,1)
        self.dWg = 0.1*np.random.randn(self.no_neurons,self.no_neurons)
        self.dbg = 0.1*np.random.randn(self.no_neurons,1)

        ''' Storing every sigmoid and tanh for evry instance of time or 
        cell since  while backpropagating each timestamp has dvalues and
        different inputs '''
        self.sigmoid_f = [Sigmoid() for _ in range(self.T)] # sigmoid for the forgetgate
        self.sigmoid_i = [Sigmoid() for _ in range(self.T)] # sigmoid for the inputgate
        self.sigmoid_o = [Sigmoid() for _ in range(self.T)] # sigmoid is for the outputgate 
        self.tanh_hc = [Tanh() for _ in range(self.T)] # thist anh is when data goes from h(t) to c(t)
        self.tanh_ch = [Tanh() for _ in range(self.T)] # this tanh is when datya comes from c(t) to h(t) 

        ht = self.H[0]
        ct = self.C[0]

        for t,xt in enumerate(self.x):
            xt = xt.reshape(1,1)

            finput = np.dot(self.Wf,ht)+np.dot(self.Uf,xt)+self.bf # Forward for forgetgate
            self.sigmoid_f[t].forward(finput)
            self.F[t] = self.sigmoid_f[t].outputs

            iinput = np.dot(self.Wi,ht)+np.dot(self.Ui,xt)+self.bi # Forward for inputgate
            self.sigmoid_i[t].forward(iinput)
            self.I[t] = self.sigmoid_i[t].outputs

            ctildainput = np.dot(self.Wg,ht)+np.dot(self.Ug,xt)+self.bg # Forwrd for c_tilda
            self.tanh_hc[t].forward(ctildainput)
            self.C_tilda[t] = self.tanh_hc[t].outputs

            oinput = np.dot(self.Wo,ht)+np.dot(self.Uo,xt)+self.bo # Forward for output gate 
            self.sigmoid_o[t].forward(oinput)
            self.O[t] = self.sigmoid_o[t].outputs

            self.C[t+1] = np.multiply(ct,self.F[t])+np.multiply(self.I[t],self.C_tilda[t]) # Forward for c(t)
            self.tanh_ch[t].forward(self.C[t+1])
            choutput = self.tanh_ch[t].outputs

            self.H[t+1] = np.multiply(choutput,self.O[t])

            # For the next iteration or cell both ct and ht has to passed to next iter or next cell
            ct = self.C[t+1] 
            ht = self.H[t+1]

    def backward(self,dvalues):

        dht = dvalues[-1,:].reshape(self.no_neurons,1) # dht for last cell is only with respect to loss and not with dht+1 is absent
        dct = np.zeros_like(self.C[-1]) 

        for t in reversed(range(self.T)):
            xt = self.x[t].reshape(1,1)

            if t==0:
                prev_H = self.H[0]
                prev_C = self.C[0]
            else:
                prev_H = self.H[t-1]
                prev_C = self.C[t-1]

            # Derivative of Uf and Wf
            self.tanh_ch[t].backward(np.multiply(dht,self.O[t]))
            dhotanch = self.tanh_ch[t].dinputs
            self.sigmoid_f[t].backward(np.multiply(dhotanch,prev_C))
            dhotanchf = self.sigmoid_f[t].dinputs
            self.dUf += np.dot(dhotanchf,xt)
            self.dWf += np.dot(dhotanchf,prev_H.T)
            self.dbf += dhotanchf

            # Derivative of Ui and Wi
            self.sigmoid_i[t].backward(np.multiply(dhotanch,self.C_tilda[t]))
            dhotanchi = self.sigmoid_i[t].dinputs
            self.dUi += np.dot(dhotanchi,xt)
            self.dWi += np.dot(dhotanchi,prev_H.T)
            self.dbi += dhotanchi

            # Derivative of Ug and Wg
            self.tanh_hc[t].backward(np.multiply(dhotanch,self.I[t]))
            dhotanchtanhc = self.tanh_hc[t].dinputs
            self.dUg += np.dot(dhotanchtanhc,xt)
            self.dWg += np.dot(dhotanchtanhc,prev_H.T)
            self.dbg += dhotanchtanhc

            # Derivative of Uo and Wo
            self.sigmoid_o[t].backward(np.multiply(dht,self.tanh_ch[t].outputs))
            dhco = self.sigmoid_o[t].dinputs
            self.dUo += np.dot(dhco,xt)
            self.dWo += np.dot(dhco,prev_H.T)
            self.dbo += dhco

            # derivative of ht-1
            dht = np.dot(self.Wf.T,dhotanchf)+np.dot(self.Wi.T,dhotanchi) \
                +np.dot(self.Wg.T,dhotanchtanhc)+np.dot(self.Wo.T,dhco) \
                +(dvalues[t-1,:].reshape(self.no_neurons,1) \
                if t >0 else np.zeros_like(dvalues[-1,:].reshape(self.no_neurons,1)))
   
class Optimizer_SGD_LSTM:
    def __init__(self, learning_rate = 1e-5, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        #if momentum
        if self.momentum:
            if not hasattr(layer, 'Uf_momentums'):
                layer.Uf_momentums = np.zeros_like(layer.Uf)
                layer.Ui_momentums = np.zeros_like(layer.Ui)
                layer.Uo_momentums = np.zeros_like(layer.Uo)
                layer.Ug_momentums = np.zeros_like(layer.Ug)
                
                layer.Wf_momentums = np.zeros_like(layer.Wf)
                layer.Wi_momentums = np.zeros_like(layer.Wi)
                layer.Wo_momentums = np.zeros_like(layer.Wo)
                layer.Wg_momentums = np.zeros_like(layer.Wg)
                
                layer.bf_momentums = np.zeros_like(layer.bf)
                layer.bi_momentums = np.zeros_like(layer.bi)
                layer.bo_momentums = np.zeros_like(layer.bo)
                layer.bg_momentums = np.zeros_like(layer.bg)
                

            Uf_updates = self.momentum * layer.Uf_momentums - \
                self.current_learning_rate * layer.dUf
            layer.Uf_momentums = Uf_updates
            
            Ui_updates = self.momentum * layer.Ui_momentums - \
                self.current_learning_rate * layer.dUi
            layer.Ui_momentums = Ui_updates
            
            Uo_updates = self.momentum * layer.Uo_momentums - \
                self.current_learning_rate * layer.dUo
            layer.Uo_momentums = Uo_updates
            
            Ug_updates = self.momentum * layer.Ug_momentums - \
                self.current_learning_rate * layer.dUg
            layer.Ug_momentums = Ug_updates
            
            Wf_updates = self.momentum * layer.Wf_momentums - \
                self.current_learning_rate * layer.dWf
            layer.Wf_momentums = Wf_updates
            
            Wi_updates = self.momentum * layer.Wi_momentums - \
                self.current_learning_rate * layer.dWi
            layer.Wi_momentums = Wi_updates
            
            Wo_updates = self.momentum * layer.Wo_momentums - \
                self.current_learning_rate * layer.dWo
            layer.Wo_momentums = Wo_updates
            
            Wg_updates = self.momentum * layer.Wg_momentums - \
                self.current_learning_rate * layer.dWg
            layer.Wg_momentums = Wg_updates
            
            bf_updates = self.momentum * layer.bf_momentums - \
                self.current_learning_rate * layer.dbf
            layer.bf_momentums = bf_updates
            
            bi_updates = self.momentum * layer.bi_momentums - \
                self.current_learning_rate * layer.dbi
            layer.bi_momentums = bi_updates
            
            bo_updates = self.momentum * layer.bo_momentums - \
                self.current_learning_rate * layer.dbo
            layer.bo_momentums = bo_updates
            
            bg_updates = self.momentum * layer.bg_momentums - \
                self.current_learning_rate * layer.dbg
            layer.bg_momentums = bg_updates
            
        else:
            
            Uf_updates = -self.current_learning_rate * layer.dUf
            Ui_updates = -self.current_learning_rate * layer.dUi
            Uo_updates = -self.current_learning_rate * layer.dUo
            Ug_updates = -self.current_learning_rate * layer.dUg
            
            Wf_updates = -self.current_learning_rate * layer.dWf
            Wi_updates = -self.current_learning_rate * layer.dWi
            Wo_updates = -self.current_learning_rate * layer.dWo
            Wg_updates = -self.current_learning_rate * layer.dWg
            
            bf_updates = -self.current_learning_rate * layer.dbf
            bi_updates = -self.current_learning_rate * layer.dbi
            bo_updates = -self.current_learning_rate * layer.dbo
            bg_updates = -self.current_learning_rate * layer.dbg
            
        
        layer.Uf += Uf_updates 
        layer.Ui += Ui_updates 
        layer.Uo += Uo_updates 
        layer.Ug += Ug_updates 
        
        layer.Wf += Wf_updates 
        layer.Wi += Wi_updates 
        layer.Wo += Wo_updates
        layer.Wg += Wg_updates
        
        layer.bf += bf_updates 
        layer.bi += bi_updates 
        layer.bo += bo_updates
        layer.bg += bg_updates
        
    def post_update_params(self):
        self.iterations += 1


            