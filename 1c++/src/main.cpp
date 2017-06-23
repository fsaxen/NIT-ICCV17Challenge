#include <iostream>
#include <experimental/filesystem>
void createFileNameList(const std::string& dataset_dir, const std::string& exdata_dir, const std::string& train_or_val_or_test);
void detectFace(const std::string& exdata_dir, const std::string& train_or_val_or_test);
int detectAUsOld(const std::string& exdata_dir, const std::string& train_or_val_or_test);
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
	

	std::cout << "1. Create Filename list from dataset folder ..." << std::endl;
 	createFileNameList(dataset_dir, exdata_dir, train_or_val_or_test);
	std::cout << "Done.\n2. Detect faces in each video ..." << std::endl;
 	detectFace(exdata_dir, train_or_val_or_test);
	std::cout << "Done.\n3. Extract Action Units in each frame ..." << std::endl;
  	detectAUsOld(exdata_dir, train_or_val_or_test); //, filename_list_filename, filename_face_detection, filename_AUsOld);
  	std::cout << "Done.\n4. Recognize faces ... " << std::endl;
  	recognizeFaces(exdata_dir, train_or_val_or_test); //, filename_list_filename, filename_face_detection, filename_face_recognition);
	std::cout << "Done. \nPlease press Enter to exit." << std::endl;
	
	std::cin.get();
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