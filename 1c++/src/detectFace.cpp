// Authors: Frerk Saxen and Philipp Werner
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
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

// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

void detectFace(const std::string& exdata_dir, const std::string& train_or_val_or_test)
{
	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();

	std::string filename_list_filename = exdata_dir + train_or_val_or_test + "_filenames.txt";
	std::string filename_face_detection = exdata_dir + train_or_val_or_test + "_facedet.txt";
	std::string net_filename = exdata_dir + "mmod_human_face_detector.dat";

	
	
	std::vector<std::string> filename_list;
	misc::read_filename_list(filename_list_filename, filename_list);

	net_type net;
	deserialize(net_filename) >> net;
	long nrVid = 0;
	long nrError = 0;

	// Open detection filename
	std::ofstream detFile(filename_face_detection);
	CV_Assert(detFile.is_open());
	
	for(const auto& filename : filename_list)
	{
		try
		{
		    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		    int seconds_expired = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count();
		    int minutes_left = nrVid == 0 ? 0 : seconds_expired * (static_cast<double>(filename_list.size() - nrVid) / (double)nrVid / 60.0);

		    // Open video
		    std::cout << rpad(cast_to_string(nrVid+1),3) << "/" << filename_list.size() << ". Approx. " << rpad(cast_to_string(minutes_left), 3) << " minutes left. " << "Processing video: " << filename;
		    cv::VideoCapture vid(filename);
		    CV_Assert(vid.isOpened());

		    cv::Mat cvBGRImg;
		    dlib::matrix<dlib::rgb_pixel> dlibRGBImg;
		    std::vector<dlib::matrix<dlib::rgb_pixel>> images;
		    dlib::rectangle det;
		    long frameCnt = 0;
		    bool vid_read = vid.read(cvBGRImg);
		    while (vid_read || !images.empty())
		    {
			    if (vid_read)
			    {
				    // Copy image to correct format
				    dlib::assign_image(dlibRGBImg, dlib::cv_image<dlib::bgr_pixel>(cvBGRImg));
				    images.push_back(std::move(dlibRGBImg));
				    vid_read = vid.read(cvBGRImg);

				    if (images.size() < 15)
					    continue;
			    }

			    // Get detections
			    auto detectionList = net(images);

			    for (auto&& dets : detectionList)
			    {
				    if (dets.empty())
				    {
					    det = dlib::rectangle();
				    }
				    else
				    {
					    // Sort detection scores in ascending order
					    std::sort(dets.begin(), dets.end(), [](const mmod_rect& left, const mmod_rect& right) {return(right.detection_confidence < left.detection_confidence); });
					    // The first detection has the highest score.
					    det = dets.at(0).rect;
				    }

				    // Write detection to file (x, y, width, height)
				    detFile << nrVid << "," << frameCnt << "," << det.left() << "," << det.top() << "," << det.width() << "," << det.height() << std::endl;

    // 				win.clear_overlay();
    // 				win.set_image(dlibBGRImg);
    // 				win.add_overlay(det);

				    frameCnt++;
			    }
			    images.clear();
		    }
		    std::cout << " with " << frameCnt << "frames. Done." << std::endl;
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
	detFile.close();
}
