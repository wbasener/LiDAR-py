import numpy as np
arrays = [np.random.randn(2, 3)for _ in range(8)]
print(np.array(arrays).shape)
print(np.stack(arrays, axis=0).shape)

print(np.stack(arrays, axis=2).shape)