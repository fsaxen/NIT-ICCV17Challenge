// Authors: Frerk Saxen and Philipp Werner
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

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
    
}