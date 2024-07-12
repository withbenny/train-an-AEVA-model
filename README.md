# Train an AEVA model

Since the original [AEVA](https://github.com/JunfengGo/AEVA-Blackbox-Backdoor-Detection-main) does not provide a code to train a model, I write this code to supplement it.

The code is based on Tensorflow 2.6.0, CudaToolkit 11.2.2, and CUDNN 8.1.0.

Notably, the model was tested with AEVA and the results obtained did not reach the results in the paper.

You also need to modify other parts of AVEA to comply with TinyImageNet.
