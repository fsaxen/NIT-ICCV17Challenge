# NIT-ICCV17Challenge
This code submission for the [ICCV 17 Real Versus Fake Expressed Emotion Challenge](http://chalearnlap.cvc.uab.es/challenge/25/track/25/description/) provides source code to extract the features and classify each video for a given folder.

If you use this code and/or our trained models, please cite:

> F. Saxen, P. Werner, and A. Al-Hamadi, ["Real vs. Fake Emotion Challenge: Learning to Rank Authenticity From Facial Activity Descriptors"](https://www.researchgate.net/publication/319316240_Real_vs_Fake_Emotion_Challenge_Learning_to_Rank_Authenticity_From_Facial_Activity_Descriptors), International Conference on Computer Vision Workshops (ICCVW), 2017.

If you use the facial action unit intensity estimation code and/or our trained models, please cite:

> P. Werner, F. Saxen, and A. Al-Hamadi, ["Handling Data Imbalance in Automatic Facial Action Intensity Estimation"](https://www.researchgate.net/publication/281811172_Handling_Data_Imbalance_in_Automatic_Facial_Action_Intensity_Estimation), British Machine Vision Conference (BMVC), 2015.

Next to the papers, see the file factsheet.pdf in the top-level directory of this repository for some details on the method etc.


## 1. System Requirements
Linux (we use Ubuntu 16.04 LTS)

## 2. Software Requirements
   * Some parts of the code are written in C++ (with full C++11 standard). We used the **gcc compiler** version (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609.
   * We use a **CMake** build process for this project. We use CMake 3.5.1 but newer versions will be just fine. Download: https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.sh
   * Other parts of the code are written in **matlab**. We used matlab version R2015a but newer and earlier versions will probably work just as fine (Octave does not provide some important features). We use mex files to train our model. Unfortunately MatlabR2015a requires gcc version 4.7. You'll have to install this version too. See 7.1 for a detailed explanation.
   * Finally the label files are generated with python code that is automatically generated in matlab. We use **python** 3.1 but python 2 works just as fine.
   * Further, you will need **cuda** 8.0 (download cuda: https://developer.nvidia.com/cuda-downloads, install help: https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04) and **cudnn** (download https://developer.nvidia.com/cudnn, you need to create an account to be able to download cuddn).

## 3. Necessary C++ Libraries
   * **dlib** version 19.4 (download http://dlib.net/files/dlib-19.4.tar.bz2)
   * **opencv** version 2.4.9  (download https://github.com/opencv/opencv/archive/2.4.9.zip)

## 4. Download exdata
Please download exdata from http://wasd.urz.uni-magdeburg.de/saxen/share/exdata.zip (455 MB)
It contains trained model files (face detector, landmark detector, action unit detector, face recognition model) that are used to extract the features. While executing the c++ and matlab code this folder will be filled with the extracted features (and some intermediate files).

## 5. Make C++ Project
We use KDevelop 4, which is internally using cmake but other IDEs should work too.
Just start CMake and link the source folder to the c++ folder and the build folder to any desired build directory (it will be automatically created).
Press configure and you will be asked to set the dlib_dir variable. Set it to the top-level dlib directory (in it there is another dlib directory and examples directory and more).
Press configure again and you should see some messages from dlib.
It is critical that dlib finds cuda and cudnn.
After the dlib is included you will be asked to set OpenCV_dir to the opencv directory.
Finally hit configure once again and generate. The c++ project should now be ready.

## 6. Execute C++ main file
Before executing the code you have to provide some arguments. The first argument is the folder location of the dataset, the second argument is the folder location of the exdata folder, and the third and last argument is either "train", "val", or "test", depending on the dataset you want to extract the features from.
E.g. to extract the testset features: "/home/user/datasets/ICCV17Challenge/Test/" "/home/user/datasets/ICCV17Challenge/exdata" "test"
If you dont want to extract the training set features, it's fine. We've provided the extracted features in exdata/AUOld_train_descriptor18.mat

## 7.1 Setup mex in MatlabR2015a
To be able to compile the SVM libraries in matlab, we use mex. Unfortunately MatlabR2015a requires gcc and g++ version 4.7. Other version of matlab do require other versions. To find out which version you need, just click on supported compilers of your version and scroll down to linux in https://de.mathworks.com/support/sysreq/previous_releases.html 
Please install gcc-4.7 and setup the matlab mex compiler:
   * sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   * sudo apt-get update
   * sudo apt-get install gcc-4.7
   * sudo apt-get install g++-4.7
   * https://stackoverflow.com/questions/8524235/how-to-provide-matlab-with-the-old-gcc-version-it-wants
   * or if ~/.matlab/R2015a/mexopts.sh does not already exist:
   * sed -e s/gcc/gcc-4.7/g /usr/local/MATLAB/R2015a/bin/mexopts.sh | sed s/g\+\+/g\+\+-4.7/g > ~/.matlab/R2015a/mexopts.sh

Hint: If you execute the 2matlab/main.m file, it will try to build the svm libraries +libSvm/private/*.mexa64. Please delete the .mexa64 files if mex failed to compile them properly.

## 7.2 Execute Matlab main file
Start matlab and open "2matlab/main.m".
You have to set the variable "data_folder" to the path of the exdata folder before you can run it.
The script imports data generated by the C++ application, generates the facial activity descriptors, runs 10-fold cross validation (without subject overlap) on the training set, trains a model on the whole training dataset, and (by default) applies it on the test dataset (and writes the results to "test_prediction.py" in the exdata folder.
If you set train_val_or_test=1, the trained model is applied on the validation set and results are written to "valid_prediction.py".

## 8. Execute automatically generated python code
To generate the pkl file that was required for submission of validation and test results, run the "valid_prediction.py" resp. "test_prediction.py" files that have been generated by the matlab code (see exdata folder).
   * python3 test_prediction.py
   * or
   * python test_prediction.py

This will generate the necessary test_prediction.pkl file.

