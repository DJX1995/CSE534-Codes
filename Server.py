import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class Server:
    np.random.seed(0)
    
    def __init__(self,in_dim, out_dim, out_dim2):
        self.X = None
        self.y = None
        self.num_examples = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_dim2 = out_dim2
        self.W = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)
        self.b = np.zeros((1, out_dim)) 
        self.z = None
        self.a = None
        self.W2 = np.random.randn(self.out_dim, self.out_dim2) / np.sqrt(self.out_dim)
        self.b2 = np.zeros((1, out_dim2)) 
        self.z2 = None
        self.a2 = None
        self.probs = None
        self.delta2 = None
        self.delta1 = None
        
    def forward(self,encrypt_x):
        # decrypt X
#         self.X = np.tanh(encrypt_x.sum(axis=1))
        self.X = np.tanh(encrypt_x)
        # forward prop
        self.num_examples = len(self.X)
        self.z = self.X.dot(self.W) + self.b
        self.X2 = np.tanh(self.z)
        # forward prop
        self.z2 = self.X2.dot(self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)           
    
    def backprop(self,y,lr=0.01,reg_lambda=0.01):
        self.y = y
        delta3 = self.probs.copy()
        delta3[range(self.num_examples), self.y] -= 1
        
        dW2 = (self.X2.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        self.delta2 = delta3.dot(self.W2.T) * (1 - np.power(self.X2, 2))
        
        dW = (self.X.T).dot(self.delta2)
        db = np.sum(self.delta2, axis=0, keepdims=True)
        self.delta1 = self.delta2.dot(self.W.T) * (1 - np.power(self.X, 2))
        
        dW2 += reg_lambda * self.W2
        self.W2 += -lr * dW2
        self.b2 += -lr * db2 
        
        dW += reg_lambda * self.W
        self.W += -lr * dW
        self.b += -lr * db   
    
    def cal_acc(self, y, i): 
        print("training acccuracy after iteration {}: {}".format(i,(np.argmax(self.probs, axis=1) == y).sum() / len(y)))
    
    def send_msg(self):
        return self.delta1