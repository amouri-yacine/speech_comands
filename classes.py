import numpy as np

y = np.load("y.npy")

classes = np.unique(y)

np.save("classes.npy", classes)

print("Classes saved:", classes)
