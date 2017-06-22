#include "misc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/data_io.h>
#include <dlib/matrix.h>
#include <vector>
#include <string>
#include <iostream>

namespace misc
{
    // This function comes from user763305 @ http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
    std::istream& safeGetline(std::istream& is, std::string& t)
    {
	t.clear();

	// The characters in the stream are read one-by-one using a std::streambuf.
	// That is faster than reading them one-by-one using the std::istream.
	// Code that uses streambuf this way must be guarded by a sentry object.
	// The sentry object performs various tasks,
	// such as thread synchronization and updating the stream state.

	std::istream::sentry se(is, true);
	std::streambuf* sb = is.rdbuf();

	for(;;) {
	    int c = sb->sbumpc();
	    switch (c) {
	    case '\n':
		return is;
	    case '\r':
		if(sb->sgetc() == '\n')
		    sb->sbumpc();
		return is;
	    case EOF:
		// Also handle the case when the last line has no line ending
		if(t.empty())
		    is.setstate(std::ios::eofbit);
		return is;
	    default:
		t += (char)c;
	    }
	}
    }
  
    void read_filename_list(const std::string& filename, std::vector<std::string>& filename_list)
    {
	    std::ifstream file(filename);
	    DLIB_CASSERT(file.is_open(), "Could not open filename list: " << filename << ".\n");

	    std::string line;
	    while (safeGetline(file, line)) 
	    {
		if(!line.empty())
		    filename_list.push_back(line);
	    }

	    file.close();
	    return;
    }
    
    void read_face_detection(const std::string& filename, std::vector<std::vector<dlib::rectangle>>& detections)
    {
	    std::ifstream file(filename);
	    DLIB_CASSERT(file.is_open(), "Could not open filename: " << filename << ".\n");

	    detections.clear();
	    std::string line;
	    int vidId, vidIdLast = 0, frameId, top, left, width, height;
	    char c;
	    std::vector<dlib::rectangle> vidDetections;
	    while (safeGetline(file, line)) 
	    {
		if(!line.empty())
		{
		    std::stringstream ss(line);
		    ss >> vidId >> c >> frameId >> c >> left >> c >> top >> c >> width >> c >> height;
		    if(vidId != vidIdLast)
		    {
			detections.push_back(vidDetections);
			vidDetections.clear();
			vidIdLast = vidId;
		    }
		    vidDetections.push_back(dlib::rectangle(left, top, left + width - 1, top + height - 1));
		}
	    }
	    // Save the last stored video data from file into vector
	    if(!vidDetections.empty())
		detections.push_back(vidDetections);

	    file.close();
	    return;
    }

    void read_face_detection(const std::string& filename, std::vector<dlib::rectangle>& detections)
    {
	    std::ifstream file(filename);
	    DLIB_CASSERT(file.is_open(), "Could not open filename: " << filename << ".\n");

	    detections.clear();
	    std::string line;
	    int top, left, width, height;
	    char c;
	    while (safeGetline(file, line)) 
	    {
		if(!line.empty())
		{
		    std::stringstream ss(line);
		    ss >> left >> c >> top >> c >> width >> c >> height;
		    detections.push_back(dlib::rectangle(left, top, left + width - 1, top + height - 1));
		}
	    }

	    file.close();
	    return;
    }
    
    
/*void read_face_detections_from_file(const std::string& filename, std::map<size_t, cv::Rect>& rect_list)
{
	rect_list.clear();
	std::ifstream file(filename);
	DLIB_CASSERT(file.is_open(), "Can not open file: " << filename);

	std::string line;
	int i, l, t, w, h; // index, left, top, width, height
	char k; // komma

	while (safeGetline(file, line))
	{
		if (line.empty())
			continue;
		std::istringstream ss(line);
		ss >> i >> k >> l >> k >> t >> k >> w >> k >> h;
		rect_list[i-1]=cv::Rect(l, t, w, h);
	}

	file.close();
	return;
}

int interpolate_coord(int y_l, int x_l, int y_r, int x_r, int x)
{
	double m = static_cast<double>(y_r - y_l) / static_cast<double>(x_r - x_l);
	return static_cast<int>(std::round(y_l + m * (x - x_l)));
}

cv::Rect interpolate_face_bbox(const cv::Rect& y_l, size_t x_l, const cv::Rect& y_r, size_t x_r, size_t x)
{
	cv::Rect y;
	y.x = interpolate_coord(y_l.x, x_l, y_r.x, x_r, x);
	y.y = interpolate_coord(y_l.y, x_l, y_r.y, x_r, x);
	y.width = interpolate_coord(y_l.width, x_l, y_r.width, x_r, x);
	y.height = interpolate_coord(y_l.height, x_l, y_r.height, x_r, x);
	return y;
}

void interpolate_face_detections(const std::pair<size_t, cv::Rect>& running_avg, std::map<size_t, cv::Rect>& face_detections)
{
	// 1. Which face was not detected? width = height = 0
	//    Get a list of detected faces (just the indices)
	std::vector<size_t> face_detected;
	for (auto&& face : face_detections)
	    if (face.second.width != 0)
		face_detected.push_back(face.first);
	  
// 	for (int i = 0; i < rect_list.size(); ++i)
// 		if (rect_list.at(i).width != 0)
// 			face_detected.push_back(i);

	// 2. If no face was found, there is nothing to interpolate
	// Fallback : Use the average face position for each frame
	if (face_detected.empty())
	{
	    for (auto&& face : face_detections)
		face.second = running_avg.second;
// 		for (int i = 0; i < rect_list.size(); ++i)
// 			rect_list.at(i) = avg_rect;
		return;
	}

	// 3. Interpolate between the detected face bounding boxes (the faces are not moving much)
	// 3.1 If the first frame was not detected, there is nothing to interpolate, we simply copy the first detected bbox to the not detected frames.
	for (size_t i = 0; i < face_detected.at(0); ++i)
		face_detections[i] = face_detections[face_detected.at(0)];

	// 3.2. The same with the last frames. 
	for (size_t i = face_detected.back() + 1; i <= face_detections.crbegin()->first; ++i)
		face_detections[i] = face_detections[face_detected.back()];

	// 3.3. The rest needs to be interpolated linearly
	for (size_t j = 0; j < face_detected.size() - 1; ++j)
	{
		size_t left_idx = face_detected.at(j);
		size_t right_idx = face_detected.at(j + 1);

		for (int i = left_idx + 1; i < right_idx; ++i)
		{
			face_detections[i] = interpolate_face_bbox(face_detections[left_idx], left_idx, face_detections[right_idx], right_idx, i);
		}
	}

	return;
}

void calc_avg_rect(const std::map<size_t, cv::Rect>& face_detections, std::pair<size_t, cv::Rect>& running_avg)
{
	dlib::matrix<double, 1, 4> sum_rect;
	sum_rect = running_avg.second.x, running_avg.second.y, running_avg.second.width, running_avg.second.height;
	sum_rect *= running_avg.first;
	size_t n2 = 0;
	for (auto&& rect : face_detections)
	{
		if (rect.second.width != 0)
		{
			sum_rect(0) += rect.second.x;
			sum_rect(1) += rect.second.y;
			sum_rect(2) += rect.second.width;
			sum_rect(3) += rect.second.height;
			++n2;
		}
	}
	sum_rect /= running_avg.first + n2;
	running_avg.first += n2;
	running_avg.second.x = std::round(sum_rect(0));
	running_avg.second.y = std::round(sum_rect(1));
	running_avg.second.width = std::round(sum_rect(2));
	running_avg.second.height = std::round(sum_rect(3));

	return;
}    
    void read_and_interpolate_face_detection(const std::string& filename, std::pair<size_t, cv::Rect>& running_avg_bbox, std::map<size_t, cv::Rect>& face_detections)
    {
	read_face_detections_from_file(filename, face_detections);
	calc_avg_rect(face_detections, running_avg_bbox);
	interpolate_face_detections(running_avg_bbox, face_detections);
    }

    int crop_face_from_bbox(const cv::Mat& input, const cv::Rect& bbox, const cv::Size& size, cv::Mat& output)
    {
	// 1. Align bbox with desired size
	// Since the detected face does not fit our desired format (size), we have to change the bbox a little.
	// We add half of the missing lines to the top and half to the bottom
	float hw_ratio = size.height / static_cast<float>(size.width);
	float desired_height = bbox.width * hw_ratio;
	float desired_width = bbox.width;
	
	float to_top = (desired_height - bbox.height) / 2.0f;
	float to_bottom = to_top;
	float to_left = 0.0f;
	float to_right = 0.0f;
	float left_over = 0.0f;
	int debug = 0;
	
	// Is the face detection partially outside the image? Let's move the bbox to the image border.
	if ( bbox.x < 0 )
	{
	    to_right = -bbox.x;
	    to_left = -to_right;
	}
	else if (bbox.x + bbox.width > input.cols - 1)
	{
	    to_left = bbox.x + bbox.width - (input.cols - 1);
	    to_right = -to_left;
	}
	
	// Do we jump out of the image coord?
	// top (no ? -> add to bottom )
	if ( (bbox.y - to_top) < 0.0f )
	{
	  to_bottom += to_top - bbox.y;
	  to_top = bbox.y;
	}
	// bottom (no? -> add to top (no? -> shrink left and right ) )
	if ( (bbox.br().y + to_bottom) > (input.rows - 1.0f) )
	{
	  left_over = (bbox.br().y + to_bottom) - (input.rows - 1.0f);
	  to_bottom = input.rows - 1.0f - bbox.br().y;
	  // Can we add the left over to the top?
	  if ( (bbox.y - to_top - left_over ) < 0.0f)
	  {
	      // This code has not been debuged yet.
	      debug = 1;
	      // no! Let's shrink left and right appropriatly
	      // But first add as much as we can to the top.
	      left_over = left_over - (bbox.y - to_top);
	      to_top = bbox.y;
	      
	      // How many pixels do we have to subtract from left and right? This depends on our new desired_height = image.rows
	      // But first let's check if everything is computed correctly
	      DLIB_CASSERT(bbox.height + to_top + to_bottom + left_over >= input.rows, "Damnit. You computed something wrong: \n  bbox.height = " << bbox.height << "\n  to_top = " << to_top << "\n  to_bottom = " << to_bottom << "  left_over = " << left_over << "\n");
	      
	      desired_height = input.rows;
	      desired_width = input.rows / hw_ratio;
	      
	      to_left = (desired_width - bbox.width) / 2.0f;
	      to_right = to_left;
	      // Both should be negative, because we want to shrink it
	      DLIB_CASSERT(to_left < 0 && to_right < 0, "Damnit. to_left = " << to_left << "  to_right = " << to_right << " \n");
	  }
	  else // Easy, there is enough room to expand the top a little bit.
	      to_top += left_over;
	}
	
	// Now, let's create our bounding box that lies within the image borders.
	cv::Rect crop(std::round(bbox.x - to_left), std::round(bbox.y - to_top), std::round(desired_width), std::round(desired_height));
	
	// Crop image
	cv::resize(input(crop),output,size);
	
	return debug;
    }
    
    void read_au_list_from_file(const std::string& filename, std::vector<Identifier>& AU_list)
    {
	    std::ifstream file(filename);
	    if (!file.is_open())
	    {
		    std::cout << "Couldn't open file: " << filename << std::endl;
		    return;
	    }

	    std::string line;
	    size_t frame_no, file_no;
	    while (std::getline(file, line))
	    {
		    std::istringstream ss(line);
		    ss >> file_no >> frame_no;

		    AU_list.push_back(Identifier(file_no, frame_no));
	    }

	    file.close();
	    return;
    }    */  
    
}