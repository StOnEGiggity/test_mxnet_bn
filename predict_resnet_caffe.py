import caffe
import numpy as np
import cv2

MEAN_COLOR = np.array([103.0626238, 115.90288257, 123.15163084]).reshape((1, 3, 1, 1)) # BGR 

deploy = "./d.prototxt"
caffe_model = "./ResNet-101-model.caffemodel"
nn = caffe.Net(deploy, caffe_model, caffe.TEST) 

x = np.arange(256 * 56 * 56).reshape((1, 256, 56, 56)).astype(np.float32)
nn.blobs["data"].data[...] = x 
nn.forward()
prob = nn.blobs["outbn"].data
print (prob.shape)

print (prob)
np.save("caffe.npy", prob)
