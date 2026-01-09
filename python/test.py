import autograd
import random
from autograd import Node
from tensor import FloatTensor

# input 1 samples of 3 features
input = FloatTensor([1,3])
input[0,0] = 1.0
input[0,1] = 0.5
input[0,2] = -1.0
x  = Node(input) 

weights1 = FloatTensor([3,4])
for i in range(3):
    for  j in range(4):
        weights1[i,j] = random.uniform(-1.0, 1.0) 
w1 = Node(weights1) 

bias = FloatTensor([1,4])
for i in range(1): 
    for j in range(4):
        bias[i, j] = random.uniform(-0.1, 0.1)
b  = Node(bias)

weights2 = FloatTensor([4,1])
for i in range(4): 
    for j in range(1):
        weights2[i,j] = random.uniform(-1.0, 1.0) 
w2 = Node(weights2) # weights l2

truth = FloatTensor([1,1])
for i in range(1): 
    for j in range (1):
        truth[i,j] = 1.0
y = Node(truth) # truth value for 1 sample

lr1 = FloatTensor(w1.data.getShape())
lr2 = FloatTensor(w2.data.getShape())
lr1.fill(-0.01)
lr2.fill(-0.01)

for i in range(50):
    h_pre = (x @ w1) + b # 1x4
    h = h_pre.relu()
    pred = h @ w2
    loss = pred.relu()

    lossGradT = FloatTensor(loss.data.getShape())
    lossGradT.fill(1.0)
    loss.grad = lossGradT
    autograd.backward(loss)

    print(loss.data.getData())

    w1.data = w1.data + lr1 * w1.grad
    w1.grad = None
    w2.data = w2.data + lr2 * w2.grad
    w2.grad = None
