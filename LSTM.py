import numpy as np

class Tanh:
    def forward(self,inputs):
        self.outputs = np.tanh(inputs)

    def backward(self,dvalues):
        derivative_of_tanh = 1-self.outputs**2
        self.dinputs = np.multiply(derivative_of_tanh,dvalues)


class Sigmoid:
    def forward(self,inputs):
        sigmoid = np.clip(1/(1+np.exp(-inputs)),1e-7,1-1e-7)
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