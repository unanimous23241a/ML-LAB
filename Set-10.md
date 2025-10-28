# Q1.
import numpy as np  
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)  
y = np.array([[92], [86], [89]], dtype=float)  
X /= np.amax(X, axis=0)  
y /= 100  

sigmoid = lambda x: 1 / (1 + np.exp(-x))  
dsigmoid = lambda x: x * (1 - x)  

epochs, lr = 7000, 0.1  
in_neurons, hid_neurons, out_neurons = 2, 3, 1  

wh = np.random.uniform(size=(in_neurons, hid_neurons))  
bh = np.random.uniform(size=(1, hid_neurons))  
wout = np.random.uniform(size=(hid_neurons, out_neurons))  
bout = np.random.uniform(size=(1, out_neurons))  

for _ in range(epochs):  
    hlayer_act = sigmoid(np.dot(X, wh) + bh)  
    output = sigmoid(np.dot(hlayer_act, wout) + bout)  
    d_output = (y - output) * dsigmoid(output)  
    d_hidden = d_output.dot(wout.T) * dsigmoid(hlayer_act)  
    wout += hlayer_act.T.dot(d_output) * lr  
    wh += X.T.dot(d_hidden) * lr  

print("Input:\n", X)  
print("Actual Output:\n", y)  
print("Predicted Output:\n", output)  
                                                                   

---

# Q2.
import matplotlib.pyplot as plt  
from sklearn.datasets import make_blobs  
from sklearn.cluster import Birch  
dataset, clusters = make_blobs(n_samples = 600, centers = 8, cluster_std = 0.75, random_state = 0)  
model = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)  
model.fit(dataset)  
pred = model.predict(dataset)  
plt.scatter(dataset[:, 0], dataset[:, 1], c = pred, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')  
plt.show()  
