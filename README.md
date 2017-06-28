# NIT-ICCV17Chaellenge
This code submission for the ICCV 17 Real Versus Fake Expressed Emotion Challenge provides source code to extract the features and classifiy each video for a given folder.

# 1. System Requirements
Linux (we use Ubuntu 16.04 LTS)

# 2. Software Requirements
Some parts of the code is written in C++ (with full C++11 standard). We used the gcc compiler version (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609.
CMake (we use a CMake build process for this project. We use CMake 3.5.1 but newer versions will be just fine. Download: https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.sh)
Other parts of the code is written in Matlab. We used matlab version R2015a but newer and earlier versions will probably work just as fine. Octave does not provide some important features.
Finally the labels are generated with automatically generated python code (from matlab). We use python 3.1

cuda 8.0 (download cuda: https://developer.nvidia.com/cuda-downloads, install help: https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)
cudnn (download https://developer.nvidia.com/cudnn, you need to create an account to be able to download cuddn).

# 3. Necessary C++ Libraries
dlib version 19.4 (download http://dlib.net/files/dlib-19.4.tar.bz2)
opencv version 3.1 or newer (download https://github.com/opencv/opencv/archive/3.1.0.zip)

# 4. Download exdata
Please download exdata from http://wasd.urz.uni-magdeburg.de/saxen/share/exdata.zip (455 MB)
It contains trained model files (face detector, landmark detector, action unit detector, face recognition model) that are used to extract the features. While executing the c++ and matlab code this folder will be filled with the extracted features (and some intermediate files).

# 5. Make Project
We use kdevelopement 4 which is interally using cmake but other IDEs should work too. Just start CMake and link the source folder to the c++ folder and the build folder to any desired build directory (it will be automatically created).
Press configure and you will be asked to set the dlib_dir variable. Set it to the top-level dlib directory (in it there is another dlib directory and examples directory and more). Press configure again and you should see some messages from dlib. It is critical that dlib finds cuda and cudnn. After the dlib is included you will be asked to set OpenCV_dir to the opencv directory. Finally hit configure once again and generate. The c++ project should now be ready.

# 6. Execute C++ main file
Before executing the code you have to provide some arguments. The first argument is the folder location of the dataset. the second argument is the folder location of the exdata folder. And the third and last argument is either "train", "val", or "test", depending on the dataset you want to extract the features from. 

# 7. Execute Matlab main file
TODO

# 8. Execute automatically generated python code
TODO

