import numpy as np
import mxnet as mx
import mxnet.autograd as ag
import mxnet.ndarray as nd

ctx = mx.cpu(0)
cudnn_off = True

data = mx.nd.load("./d.params")
mmean = data['aux:bn2a_branch1_moving_mean'].reshape((1, 256, 1, 1))
mvar = data['aux:bn2a_branch1_moving_var'].reshape((1, 256, 1, 1))
beta = data['arg:bn2a_branch1_beta'].reshape((1, 256, 1, 1))
gamma = data['arg:bn2a_branch1_gamma'].reshape((1, 256, 1, 1))

x = mx.nd.arange(256 * 56 * 56).reshape((1, 256, 56, 56)).astype(np.float32)
f = 1.0 / mx.nd.sqrt(mvar + 1e-5)

USING_DIV = True

if USING_DIV:
    w = (x - mmean) / mx.nd.sqrt(mvar + 1e-5)
else:
    w = (x - mmean) * f

y = gamma * w + beta

ty = np.load("./caffe.npy")


h = lambda x : mx.nd.array(x, ctx = ctx)
hx = h(x)
hx.attach_grad()

with ag.record():
    my, mxmean, mxvar = mx.nd.BatchNorm(data = hx, gamma = h(gamma), beta = h(beta), moving_mean = h(mmean), moving_var = h(mvar), eps = 1e-5, fix_gamma = False, use_global_stats = False, output_mean_var = True, cudnn_off = cudnn_off)

my = mx.nd.BatchNorm(data = hx, gamma = h(gamma), beta = h(beta), moving_mean = h(mmean), moving_var = h(mvar), eps = 1e-5, fix_gamma = False, use_global_stats = True, output_mean_var = False, cudnn_off = cudnn_off)

ty = nd.array(ty, ctx = ctx)

print ("context: ", ctx)
print ("cudnn_off: ", cudnn_off)
if USING_DIV:
    print ("ndarray: y = gamma * [(x - mean) / sqrt(var + eps)] + beta")
else:
    print ("ndarray: y = gamma * [(x - mean) * (1.0 / sqrt(var + eps)) ] + beta")

print ("")

print ("                max absolute error, max relative error, mean error")

print ("caffe and ndarray", nd.max(nd.abs(ty - y)).asscalar(), nd.max(nd.abs((ty - y) / y)).asscalar(), nd.mean(nd.abs(ty - y)).asscalar())

print ("caffe and mx", nd.max(nd.abs(ty - my)).asscalar(), nd.max(nd.abs((ty - my) / y)).asscalar(), nd.mean(nd.abs(ty - my)).asscalar())
print ("ndarray and mx", nd.max(nd.abs(y - my)).asscalar(), nd.max(nd.abs((y - my) / y)).asscalar(), nd.mean(nd.abs(y - my)).asscalar())
