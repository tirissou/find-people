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
=======
We tried three approaches.
First, we tried to use a native C++ model and adapt it to our dataset. Its broken.
Second, we made a model in python, exported it to C++ tensorflow and it works fast. Really fast.
Third, we tried to compare that with our original model in python, which is the python script installed. For python, we used the YOLOv1 algorithm, which worked better than our C++ box counterpart. 

# Dependencies:
 1. Native CNN:
  a. CMAKE
  b. Armadillo
  c. Boost
 2. C++ Model
  a. Bazel
  b. Tensorflow (use the tensorClone.sh shell script)
  c. openCV
  *note- you have to clone tensorflow into this repo, then mv WORKSPACE and the cpp model to tensorflow/cc/examples
  otherwise, it wont run.*
  3. Python
  a. Tensorflow
  b. Keras
  c. Scipy
  d. PIL
  e. Yad2k
  
# BenchMarks:
![alt text](https://github.com/tirissou/find-people/blob/master/oops.jpg)

# The dataset
The dataset that we used for most of the training until Austin finished his data compilation is here.
Download it and move the files into this directory.
http://pascal.inrialpes.fr/data/human/

Austin, please put the link to our total dataset here

# Running the prediction:
For each model, run the following:
 ./[insert model exec file name]  --image <image path>
  
  # To do
  - fix native CNN implementation
  - make it so you dont have to clone TF
  - Fix python bugs on runtime
