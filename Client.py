import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class Client:
    
    def __init__(self,data_x,data_y,out_dim):
        np.random.seed(1000)
        self.X = data_x
        self.y = data_y
        self.num_examples = len(self.X)
        self.in_dim = self.X.shape[1]
        self.out_dim = out_dim
        self.W = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)
        self.b = np.zeros((1, out_dim)) 
        self.b = np.random.rand(1, out_dim)
        self.z = None
        self.a = None
        self.true_z = None
        self.encrypt_z = None
        
    def forward(self):
        self.z = self.X.dot(self.W) + self.b
        self.a = np.tanh(self.z) 
#         print('encript start')
#         self.encrypt_forward()
#         print('encript end')
    
    def backprop(self,delta,lr=0.01,reg_lambda=0.01):
        dW = np.dot(self.X.T, delta)
        db = np.sum(delta, axis=0)
        dW += reg_lambda * self.W
        self.W += -lr * dW
        self.b += -lr * db       
    
    def encrypt_forward(self):
        res1 = []
        res2 = []
        for i in range(self.X.shape[0]):
            np.random.seed(i)
            random_key = np.random.rand(1,self.out_dim)
            temp = self.X[i].reshape(-1,1) * self.W
            temp = np.append(temp,self.b,axis=0)
#             temp2 = temp + np.concatenate((2*random_key,-random_key,-random_key),axis=0)
            temp2 = temp
            res1.append(temp)
            res2.append(temp2)
        self.true_z = np.array(res1)
        self.encrypt_z = np.array(res2)
    
    def send_msg(self):
        return self.z