#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <chrono>

// Authors: Frerk Saxen and Philipp Werner
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <FaceBase/FaceRegistrationAffineMeanShape.hpp>
#include <FaceBase/FaceLibDlib.hpp>

#include <ActionUnitIntensityEstimation/AU.hpp>

#include "misc.hpp"

using namespace dlib;
using namespace std;

int detectAUsOld(const std::string& exdata_dir, const std::string& train_or_val_or_test)
try
{
	std::string filename_list_filename = exdata_dir + train_or_val_or_test + "_filenames.txt";
	std::string filename_face_detection = exdata_dir + train_or_val_or_test + "_facedet.txt";
	std::string filename_AUsOld = exdata_dir + train_or_val_or_test + "_AUOld.txt";
	
	std::string shape_predictor_file = exdata_dir + "spd+all=cascade30+oversampling70+trees1500.dat";
	std::string mean_shape_file = exdata_dir + "mean_face_shape_intraface.dat";
	std::string feature_model_file = exdata_dir + "lbp_10_10_8_1.txt";
	std::string mean_std_file = exdata_dir + "Features_mean_std_disfa.txt";
	std::string regressor_file = exdata_dir + "Regression_model_disfa_1_2_4_6_9_12_25.txt";

	 
	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();


	std::vector<std::string> filename_list;
	misc::read_filename_list(filename_list_filename, filename_list);

	
	std::vector<std::vector<dlib::rectangle>> face_dets_list;
	misc::read_face_detection(filename_face_detection, face_dets_list);

	
	using t_AUs = std::vector<float>;
	using t_AU_vid = std::vector<t_AUs>;
	using t_AU_vids = std::vector<t_AU_vid>;
	t_AU_vids au_vids;

	DLIB_CASSERT(filename_list.size() == face_dets_list.size(), "List size mismatch: \n\t filename_list.size(): " << filename_list.size() << "\n\t face_dets_list.size(): " << face_dets_list.size() << std::endl);
	

	shape_predictor sp;
	deserialize(shape_predictor_file) >> sp;

	
	FaceRegistrationAffineMeanShape face_reg;
	DLIB_CASSERT(face_reg.init(mean_shape_file.c_str(), cv::Size(200, 200), 0.5),  "Error loading/initializing the face registration model: " << mean_shape_file);

	AUIntensityEstimation AU(feature_model_file, mean_std_file, regressor_file);
	DLIB_CASSERT(AU.is_initialized(), "Error loading AU model.");

	

	matrix<rgb_pixel> img;
	cv::Mat cvImage, face_registered, AU_detections;
	std::vector<cv::Point2f> landmarks68, landmarks49, landmarks49_registered;
	std::vector<int> AU_IDs; // Empty vector -> All AUs are going to become visualized
	cv::Rect bbox;
	t_AUs au;
	t_AU_vid AU_vid;

	for(long vid_id = 0; vid_id < filename_list.size(); ++vid_id)
	{
	      const auto vid_filename = filename_list.at(vid_id);
	      
	      std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	      int seconds_expired = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count();
	      int minutes_left = vid_id == 0 ? 0 : seconds_expired * (static_cast<double>(filename_list.size() - vid_id) / (double)vid_id / 60.0);
	      std::cout << rpad(cast_to_string(vid_id+1),3) << "/" << filename_list.size() << ". Approx. " << rpad(cast_to_string(minutes_left), 3) << " minutes left. " << "Processing video: " << vid_filename << std::endl;
	  
	      cv::VideoCapture vid(vid_filename);
	      DLIB_CASSERT(vid.isOpened(), "Cannot open video filename : " << vid_filename);
	      
	      const std::vector<dlib::rectangle>& face_dets = face_dets_list.at(vid_id);
	      
	      long frame_no = 0;
	      AU_vid.clear();
	      while (vid.read(cvImage))
	      {
		      // Prepare for next frame
		      dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(cvImage));

		      const auto& face_det = face_dets.at(frame_no);
		      
		      // Get landmarks
		      dlib::full_object_detection shape = sp(img, face_det);
		      matrix<rgb_pixel> face_chip;
		      // 1. From dlib to opencv
		      landmarks68.clear();
		      for (long part_no = 0; part_no < shape.num_parts(); ++part_no)
			      landmarks68.push_back(cv::Point2f(shape.part(part_no).x(), shape.part(part_no).y()));
		      // 2. From 68 to 49 (inner landmarks)
		      FaceLibDlib::conv_landmarks_68_to_49(landmarks68, landmarks49);

		      // Register face
		      face_reg.register_face(landmarks49, cvImage, &face_registered, &landmarks49_registered);

		      // Estimate AU Intensity
		      AU.estimate(face_registered, landmarks49_registered, AU_detections);

		      // Visualize AUs
		      // 1. Convert bounding box into opencv format
		      //bbox = cv::Rect(std::max(face_det.left(),(long)0), std::max(face_det.top(),(long)0), face_det.width(), face_det.height());
		      //AU.visualize(cvImage, landmarks49, AU_detections, AU_IDs, bbox);
		      
		      au.clear();
		      for(int auIdx = 0; auIdx < AU_detections.cols; ++auIdx)
			  au.push_back(AU_detections.at<float>(auIdx));
		      AU_vid.push_back(au);
  
		      ++frame_no;
		      //cv::imshow("frame", cvImage);

	      }
	      au_vids.push_back(AU_vid);
	}
	
	
	// Save AUs to file
	DLIB_CASSERT(au_vids.size() == filename_list.size(), "au_vids.size() != filename_list.size(). \n\t au_vids.size(): " << au_vids.size() << "\n\t filename_list.size(): " << filename_list.size() << std::endl);
	// Open AU filename
	std::ofstream AUFile(filename_AUsOld);
	DLIB_CASSERT(AUFile.is_open());
	for(long vid_id = 0; vid_id < au_vids.size(); ++vid_id)
	    for(long frame_no = 0; frame_no < au_vids.at(vid_id).size(); ++frame_no)
	    {
		    AUFile << vid_id << "," << frame_no;
		    for(int auIdx = 0; auIdx < au_vids.at(vid_id).at(frame_no).size(); ++auIdx)
			AUFile << "," << au_vids[vid_id][frame_no][auIdx];
		    AUFile << "\n";
	    }
	AUFile.close();
	
	return 0;
}
catch (std::exception& e)
{
	cout << e.what() << endl;
	//cout << "hit enter to terminate" << endl;
	//cin.get();
	return -1;
}
