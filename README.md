# Test MXNet BatchNorm

- caffe2mx.py
    Convert the caffe model into *.params

    The download link of ResNet-101-model.caffemodel is [here](https://github.com/KaimingHe/deep-residual-networks)

    I have converted it to d.params

- predict_resnet_caffe.py
   Test the BN layer in Caffe and the output is saved as caffe.npy   

- compute.py 
    Test the BN layer in MXNet and compare the outputs of MXNet, Caffe and numpy
