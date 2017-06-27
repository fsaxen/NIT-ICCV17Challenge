// Authors: Frerk Saxen and Philipp Werner (Frerk.Saxen@ovgu.de, Philipp.Werner@ovgu.de)
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

/* This is the main execution entry for extracting all necessary features for training, validation, and testing.
 * After providing the necessary arguments (see function help() ), the following functions will be executed in order:
 * 1. createFileNameList(): A filename list txt file will be created in the exdata directory that lists all video filenames in the dataset folder (given as argument).
 * 2. detectFace(): For each frame of all videos the face will be detected and the results will be stored in a xxx_facedet.txt file.
 * 3. detectAUsOld(): We extract 7 different facial action units for each frame and save the results to another txt file.
 * 4. recognizeFaces(): We cluster similar faces in the dataset to allow intra-personal classification.
 */
#include <iostream>
#include <experimental/filesystem>
void createFileNameList(const std::string& dataset_dir, const std::string& exdata_dir, const std::string& train_or_val_or_test);
void detectFace(const std::string& exdata_dir, const std::string& train_or_val_or_test);
void detectAUsOld(const std::string& exdata_dir, const std::string& train_or_val_or_test);
void recognizeFaces(const std::string& exdata_dir, const std::string& train_or_val_or_test);
int help();


namespace fs = std::experimental::filesystem;

int main(int argc, char **argv) 
{
	if(argc != 4)
	      return help();
	
	std::string dataset_dir = std::string(argv[1]);
	std::string exdata_dir = std::string(argv[2]);
	std::string train_or_val_or_test = std::string(argv[3]);
	
	if(!fs::is_directory(exdata_dir))
	{
		std::cout << "Error: " << exdata_dir << " is not a valid directory." << std::endl;
		return -1;
	}
	if(dataset_dir.back() != '/')
		dataset_dir.push_back('/');
	if(exdata_dir.back() != '/')
		exdata_dir.push_back('/');
	
	try
	{
		std::cout << "1. Create Filename list from dataset folder ..." << std::endl;
		createFileNameList(dataset_dir, exdata_dir, train_or_val_or_test);
		std::cout << "Done.\n2. Detect faces in each video ..." << std::endl;
		detectFace(exdata_dir, train_or_val_or_test);
		std::cout << "Done.\n3. Extract Action Units in each frame ..." << std::endl;
		detectAUsOld(exdata_dir, train_or_val_or_test); //, filename_list_filename, filename_face_detection, filename_AUsOld);
		std::cout << "Done.\n4. Recognize faces ... " << std::endl;
		recognizeFaces(exdata_dir, train_or_val_or_test); //, filename_list_filename, filename_face_detection, filename_face_recognition);
 		std::cout << "Done. \nYou are now finished with the C++ part. Please execute the main.m file in the matlab folder with matlab R2015a or newer.\nPress Enter to continue." << std::endl;
		std::cin.get();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		std::cout << "Caution. Errors occured. You can not proceed with matlab. Please contact the authors if you cannot resolve the issues. Press enter to terminate." << std::endl;
		std::cin.get();
		return -1;
	}	
	return 0;
}

int help()
{
	std::cout << std::endl;
	std::cout << "usage: NIT-ICCV17Challenge <dataset_dir> <exdata_dir> <train_or_val_or_test>" << std::endl;
	std::cout << "dataset_dir: Please provide the full folder name of the (training, validation, or testing) video dataset. E.g. /home/user/iccvdataset/test/" << std::endl;
	std::cout << "exdata_dir: This program extracts the features for the matlab traning and testing procedure. Please provide a folder for this data. E.g. /home/user/iccvdataset/exdata/" << std::endl;
	std::cout << "train_or_val_or_test: Since we extract training, validation, or testing data, please provide a postfix to identify with either {train, val, test} depending on the execution." << std::endl;
	std::cout << std::endl;
	return -1;
}