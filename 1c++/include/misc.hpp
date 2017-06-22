#pragma once

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <dlib/geometry.h>

namespace misc
{
    // This function comes from user763305 @ http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
    std::istream& safeGetline(std::istream& is, std::string& t);
  
    void read_filename_list(const std::string& filename, std::vector<std::string>& filename_list);
    
    void read_face_detection(const std::string& filename, std::vector<std::vector<dlib::rectangle>>& detections);
    void read_face_detection(const std::string& filename, std::vector<dlib::rectangle>& detections);
//     int crop_face_from_bbox(const cv::Mat& input, const cv::Rect& bbox, const cv::Size& size, cv::Mat& output);
//     
//     struct Identifier
//     {
// 	    Identifier(size_t _file_no, size_t _frame_no) : file_no(_file_no), frame_no(_frame_no)  {};
// 	    size_t  file_no, frame_no;
//     };
//     void read_au_list_from_file(const std::string& filename, std::vector<Identifier>& AU_list);
      
}