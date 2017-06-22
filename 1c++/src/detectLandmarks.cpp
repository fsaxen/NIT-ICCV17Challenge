#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/videoio/videoio.hpp>
#include <fstream>
#include <string>
#include <chrono>
#include "misc.hpp"

//using namespace std;
using namespace dlib;


void detectLandmarks()
{
	std::string iccvDatasetFolder = "/home/frerk/datasets/ICCV17Challenge/";

 	std::string srcFolder = iccvDatasetFolder;
	std::string destfile = iccvDatasetFolder + "exdata/landmarks_train.txt";
	std::string filename_list_file = iccvDatasetFolder + "exdata/filenames_train.txt";
	std::string filename_face_det_file = iccvDatasetFolder + "exdata/facedet_train.txt";
	std::string landmark_model_filename = iccvDatasetFolder + "exdata/spd+all=cascade30+oversampling70+trees1500.dat";

	
	
	std::vector<std::string> filename_list;
	std::vector<std::vector<dlib::rectangle>> facedet_list;
	misc::read_filename_list(filename_list_file, filename_list);
	misc::read_face_detection(filename_face_det_file, facedet_list);

	long nrVid = 0;
	long nrError = 0;

	// Open detection filename
	std::ofstream lmFile(destfile);
	CV_Assert(lmFile.is_open());
	
	dlib::shape_predictor lm_model;
	deserialize(landmark_model_filename) >> lm_model;
	
	//dlib::image_window win;
	
	dlib::rand rnd(time(0));

// 	while(true)
	for(const auto& filename : filename_list)
	{
	      
		try
		{
// 		    nrVid = rnd.get_random_32bit_number() % facedet_list.size();
// 		    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
// 		    int seconds_expired = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count();
// 		    int minutes_left = nrVid == 0 ? 0 : seconds_expired * (static_cast<double>(filename_list.size() - nrVid) / (double)nrVid / 60.0);

		    // Open video
// 		    std::cout << rpad(cast_to_string(nrVid),3) << "/" << filename_list.size() << ". Approx. " << rpad(cast_to_string(minutes_left), 3) << " minutes left. " << "Processing video: " << filename;
		    cv::VideoCapture vid(srcFolder + filename_list.at(nrVid));
		    CV_Assert(vid.isOpened());

		    cv::Mat cvBGRImg;
		    dlib::matrix<dlib::rgb_pixel> dlibRGBImg;
		    
		    long frameCnt = 0;
		    while (vid.read(cvBGRImg))
		    {
			    // Copy image to correct format
			    dlib::assign_image(dlibRGBImg, dlib::cv_image<dlib::bgr_pixel>(cvBGRImg));

			    // Get landmarks
 			    dlib::full_object_detection shape = lm_model(dlibRGBImg, facedet_list.at(nrVid).at(frameCnt));

			    lmFile << nrVid << "," << frameCnt;
			    for(int partNr = 0; partNr < shape.num_parts(); ++partNr)
			    {
				  dlib::point p = shape.part(partNr);
				  lmFile << "," << p.x() << "," << p.y();
			    }
			    lmFile << "\n";
// 			    std::vector<dlib::full_object_detection> shapes;
// // 			    shapes.push_back(shape);
// 			    win.clear_overlay();
// 			    win.set_image(dlibRGBImg);
// 			    win.add_overlay(facedet_list.at(nrVid).at(frameCnt), dlib::rgb_pixel(0,0,255));
//  			    win.add_overlay(dlib::render_face_detections(shape));
// 			    dlib::sleep(1);

			    frameCnt++;
		    }
		    
 		    std::cout << "nrVid: " << nrVid << " with " << frameCnt << "frames. Done." << std::endl;
		    ++nrVid;
		}
		catch(const std::exception& e)
		{
		    std::cout << "\n-----------------------------------------------------------------------\n";
		    std::cout << "Error at vid nr: " << nrVid << std::endl;
		    std::cout << e.what() << std::endl;
		    std::cout << "-----------------------------------------------------------------------\n\n";
		    ++nrVid;
		    ++nrError;
		}
		catch(...)
		{
		    std::cout << "\n-----------------------------------------------------------------------\n";
		    std::cout << "Error at vid nr: " << nrVid << std::endl;
		    std::cout << "-----------------------------------------------------------------------\n\n";
		    ++nrVid;
		    ++nrError;
		}
	}
	
	std::cout << "Finished. Number of Errors: " << nrError << std::endl;
 	lmFile.close();
}
