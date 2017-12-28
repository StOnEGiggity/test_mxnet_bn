import numpy as np
import mxnet as mx
import mxnet.autograd as ag

data = mx.nd.load("./d.params")
mmean = data['aux:bn2a_branch1_moving_mean'].asnumpy().reshape((1, 256, 1, 1))
mvar = data['aux:bn2a_branch1_moving_var'].asnumpy().reshape((1, 256, 1, 1))
beta = data['arg:bn2a_branch1_beta'].asnumpy().reshape((1, 256, 1, 1))
gamma = data['arg:bn2a_branch1_gamma'].asnumpy().reshape((1, 256, 1, 1))

x = np.arange(256 * 56 * 56).reshape((1, 256, 56, 56)).astype(np.float32)
f = 1.0 / np.sqrt(mvar + 1e-5)

w = (x - mmean) / np.sqrt(mvar + 1e-5)
# w = (x - mmean) * f

y = gamma * w + beta

ty = np.load("./caffe.npy")

h = mx.nd.array
hx = h(x)
hx.attach_grad()

with ag.record():
    my, mxmean, mxvar = mx.nd.BatchNorm(data = hx, gamma = h(gamma), beta = h(beta), moving_mean = h(mmean), moving_var = h(mvar), eps = 1e-5, fix_gamma = False, use_global_stats = False, output_mean_var = True)

my = mx.nd.BatchNorm(data = hx, gamma = h(gamma), beta = h(beta), moving_mean = h(mmean), moving_var = h(mvar), eps = 1e-5, fix_gamma = False, use_global_stats = True, output_mean_var = False)

my = my.asnumpy()
mxmean = mxmean.asnumpy()
mxvar = mxvar.asnumpy()

print ("caffe and numpy", np.max(np.abs(ty - y)), np.max(np.abs((ty - y) / y)) )

print ("caffe and mx", np.max(np.abs(ty - my)), np.max(np.abs((ty - my) / y)) ) 
print ("numpy and mx", np.max(np.abs(y - my)), np.max(np.abs((y - my) / y)) )

# print ("diff mean", np.max(np.abs(mxmean - np.mean(x, (0,2,3)))))
# print ("diff var", np.max(np.abs(mxvar - np.var(x, (0,2,3)) )))
