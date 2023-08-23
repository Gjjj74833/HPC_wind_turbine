import numpy as np
import sys

x = np.array([0, 2, 4])
print(sys.argv[1])
np.savez(f'./results/{sys.argv[1]}.npz', x = x)
