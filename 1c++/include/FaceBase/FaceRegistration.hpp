#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

/// Base class for face registration methods
class FaceRegistration
{
	std::string method_name;
public:
	/// Constructor
	FaceRegistration(const char * name) : method_name(name) {}

	/// Get name of library
	const char * get_name() const { return method_name.c_str(); }

	/// Register face, needs landmarks coordinates inside image, returns transformed image and/or landmarks if needed
	virtual bool register_face(const std::vector<cv::Point2f> & in_landmarks, const cv::Mat & in_image, cv::Mat * transformed_image = NULL, std::vector<cv::Point2f> * transformed_landmarks = NULL) = 0;

	/// Visualize landmarks (may include additional information like mean shape)
	virtual bool visualize_landmarks(cv::Mat & image, const std::vector<cv::Point2f> & landmarks) { return false; }
};
