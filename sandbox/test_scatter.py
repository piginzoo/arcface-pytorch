import matplotlib.pyplot as plt
import numpy as np


x=[1,2,3,4,5,6]
y=[13,12,41,45,2,21]
plt.scatter(x,y)
plt.legend(np.arange(10, dtype=np.int32))
plt.show()

# python test_scatter.py