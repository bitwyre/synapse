# Synapse: A Efficient Inference Library for Onnx in Native C++/C

## Abstract

Often during a startup creation we need fast experimentation and also fast implementation to production.
While that is the ideal, pushing research code into production often is not easy as it sounds. Right now
the availability of Pytorch as a fast deep learning framework allows researchers to buiild models easily
while debugging them fast. Using Pytorch you can export your models in Onnx format and then load them in 
Caffe2 or Tensorflow to do inference. Although if means in production you need to load Caffe2 and Tensorflow
in memory. This I could be much more efficient by loading the model architecture in Onnx format, load the 
model, then conduct the tensor operations in native C++/C format, optimized for each arhictecture using 
SIMD, OpenCL or CUDA code. This research has been started during the author's time at the Montreal
Institute of Learning Algorithms (MILA)/Laboratoire d’Informatique des Systèmes Adaptatifs (LISA)
inspired by implementing RNNs/LSTMs in production.

## Copyright

Dendi Suhubdy
