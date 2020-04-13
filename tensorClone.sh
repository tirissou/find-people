#!/bin/sh

repository="https://github.com/tensorflow/tensorflow.git"
echo "You are about to clone Tensorflow into your current Repo. It better be Find-People."

location=$(pwd)
newFolder="externLibs"
mkdir ${newFolder}
cd ${newFolder}
pwd
newFolderPlus="/externLibs"

git init
git remote add origin ${location}${newFolderPlus}
git fetch
git checkout -t origin/master


git clone $repository 
