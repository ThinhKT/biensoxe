import bitstring
import numpy as np
f1 = bitstring.BitArray(float = 9.000000000000000000e+01, length = 32)
print(np.asarray(f1, dtype = np.float32).view(np.int32).item())
input()
