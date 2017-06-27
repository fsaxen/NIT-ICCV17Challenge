// Authors: Frerk Saxen and Philipp Werner (Frerk.Saxen@ovgu.de, Philipp.Werner@ovgu.de)
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <dlib/string.h>

namespace fs = std::experimental::filesystem;

bool is_valid_ext(const fs::path& path);

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
		{
			for(auto& f2: fs::directory_iterator(f))
				if(is_valid_ext(f2.path()))
				{
					oFile << f2.path().string() << '\n';
				}
		}
		else
		{
			if(is_valid_ext(f.path()))
				oFile << f.path().string() << '\n';
		}
	}
	oFile.close();
}

bool is_valid_ext(const fs::path& path)
{
	std::vector<std::string> valid_file_ext = {".mp4",".avi"};
	std::string ext = dlib::tolower(path.extension().string());
	
	
	for(auto val_ext : valid_file_ext)
		if(ext.compare(val_ext) == 0)
			return true;
	return false;
}
