#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <dlib/string.h>

namespace fs = std::experimental::filesystem;

void createFileNameList(const std::string& dataset_dir, const std::string& exdata_dir, const std::string& train_or_val_or_test)
{
	std::string filename_list_filename = exdata_dir + train_or_val_or_test + "_filenames.txt";
	
	if(!fs::is_directory(dataset_dir))
	{
		std::cout << "Error: " << dataset_dir << " is not a valid directory." << std::endl;
		return;
	}

	// Open detection filename
	std::ofstream oFile(filename_list_filename);
	DLIB_CASSERT(oFile.is_open(), "Error: Could not open " << filename_list_filename << " for writing." << std::endl);
	
	for(auto& f: fs::directory_iterator(dataset_dir))
	{
		if(fs::is_directory(f))
			for(auto& f2: fs::directory_iterator(f))
				oFile << f2.path().string() << '\n';
		else
			oFile << f.path().string() << '\n';
	}
	oFile.close();
}