# find-people
Image Segmentation and CNN Implementation to Detect Humans


To Use the Tensorflow CNN model, we will be using Bazel. Bazel is like make, but its what tensorflow originally compiles into. The installation instructions are here. https://docs.bazel.build/versions/master/install.html

After installing bazel, you need to have tensorflow to link:
These commands will do it:
mkdir /path/tensorflow
cd /path/tensorflow
git clone https://github.com/tensorflow/tensorflow.git

CNN architecture:
CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED-> SOFTMAX




List of features to implement:
Convolutional 2d layer function
Relu function on nodes
Maxpool function
Flatten Function
Fully Connected Layer functions
softmax

An object called "model" that can have all of those functions parse through it at the same time. 
