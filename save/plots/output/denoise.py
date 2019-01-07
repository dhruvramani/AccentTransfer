import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=np.nan)
#inp = np.load("./input_np.dat")
op = np.load("./output_np.dat")

print(inp.shape, op.shape)
for i in range(0, op.shape[1] - 10, 10):
    sub = op[:, i:i+10]
    matplotlib.image.imsave('./plots/{}.png'.format(i), sub)
    print("Image {} saved".format(i))

