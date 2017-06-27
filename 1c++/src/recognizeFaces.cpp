// Authors: Frerk Saxen and Philipp Werner (Frerk.Saxen@ovgu.de, Philipp.Werner@ovgu.de)
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

//#include <FaceBase/FaceRegistrationTrained.hpp>
//#include <FaceBase/FaceLibDlib.hpp>

#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

#include "misc.hpp"

using namespace dlib;
using namespace std;

// ----------------- Face Recognition CNN Architecture --------------------------------------------
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using face_rec_net_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

void recognizeFaces(const std::string& exdata_dir, const std::string& train_or_val_or_test)
{
	std::string filename_list_filename = exdata_dir + train_or_val_or_test + "_filenames.txt";
	std::string filename_face_detection = exdata_dir + train_or_val_or_test + "_facedet.txt";
	std::string filename_face_recognition = exdata_dir + train_or_val_or_test + "_face_recognition.txt";

	
	std::string shape_predictor_file = exdata_dir + "spd+all=cascade30+oversampling70+trees1500.dat";
	std::string face_recognition_file = exdata_dir + "dlib_face_recognition_resnet_model_v1.dat";
	 
	const long max_frames = 50;
	
	
	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();

	std::vector<std::string> filename_list;
	misc::read_filename_list(filename_list_filename, filename_list);

	std::vector<std::vector<dlib::rectangle>> face_dets_list;
	misc::read_face_detection(filename_face_detection, face_dets_list);

	
	DLIB_CASSERT(filename_list.size() == face_dets_list.size(), "List size mismatch: \n\t filename_list.size(): " << filename_list.size() << "\n\t face_dets_list.size(): " << face_dets_list.size() << std::endl);
	

	shape_predictor sp;
	deserialize(shape_predictor_file) >> sp;

	face_rec_net_type face_rec_net;
	deserialize(face_recognition_file) >> face_rec_net;
	
	long nrError = 0;

	matrix<dlib::rgb_pixel> img;
	cv::Mat cvImage;
	typedef matrix<float, 0, 1> sample_type;
	std::vector<std::vector<sample_type>> face_descriptors;
	std::vector<matrix<rgb_pixel>> first_frame;

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
	      std::vector<matrix<rgb_pixel>> faces;
	      matrix<rgb_pixel> face_chip;
	      
	      long frame_no = 0;
	      while (vid.read(cvImage) && frame_no / 4.0 < max_frames)
	      {
		      // Take every 4th frame
		      if(frame_no % 4 != 0)
		      {
			      ++frame_no;
			      continue;
		      }
		      
		      // Prepare for next frame
		      dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(cvImage));
		      
		      const auto& face_det = face_dets.at(frame_no);
		      
		      // Get landmarks and face chip
		      auto shape = sp(img, face_det);
		      auto face_details = get_face_chip_details(shape, 150, 0.25);
		      extract_image_chip(img, face_details, face_chip); //, 150, 0.25
 		      faces.push_back(move(face_chip));
		      
		      ++frame_no;
	      }
	      first_frame.push_back(faces.at(0));
	      
	      // Perform face recognition
	      face_descriptors.push_back(face_rec_net(faces));
	}
	
// 	dlib::deserialize(exdata_dir + "train_face_descriptors.dat") >> face_descriptors >> first_frame;
	
	
	typedef std::vector<sample_type> vec_sample_type;
	
	const long num_clusters = filename_list.size() / 12; // There are always 12 videos of the same person
	
	auto dist_function = [](const vec_sample_type& a, const vec_sample_type&b)
	{
		std::vector<double> dists;
		for (size_t i = 0; i < a.size(); ++i)
		    for (size_t j = 0; j < b.size(); ++j)
			dists.push_back(length(a[i]-b[j]));
		std::sort(dists.begin(),dists.end());
 		double median = dists.at(dists.size() / 2);
		return 1.0 / (median + 0.00001);
	};
	
 	std::vector<unsigned long> labels = spectral_cluster(dist_function, face_descriptors, num_clusters);
	
	
	// Now let's display the face clustering results on the screen.  It hopefully
	// correctly grouped all the faces. 
	std::vector<image_window> win_clusters(num_clusters);
	for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
	{
	    std::vector<matrix<rgb_pixel>> temp;
	    for (size_t j = 0; j < labels.size(); ++j)
	    {
		if (cluster_id == labels[j])
		    temp.push_back(first_frame[j]);
	    }
	    win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
	    win_clusters[cluster_id].set_image(tile_images(temp));
	}
	
	// Open dest filename
	std::ofstream idFile(filename_face_recognition);
	DLIB_CASSERT(idFile.is_open());
	DLIB_CASSERT(labels.size() == filename_list.size());
	for(long vid_id = 0; vid_id < filename_list.size(); ++vid_id)
	{
		  idFile << vid_id << "," << labels[vid_id] << "\n";
	}
	idFile.close();
//  	std::cin.get();
	
	return;
}