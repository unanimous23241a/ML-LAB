# Q1.
To be added
                                                                   

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
