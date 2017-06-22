#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

/// Base class for face libraries (landmark detection and/or head pose estimation)
class FaceLib
{
	std::string lib_name;
public:
	/// Constructor
	FaceLib(const char * name) : lib_name(name) {}

	/// Get name of library
	const char * get_name() const { return lib_name.c_str(); }

	/// Process image
	/// Caution: This method is deprecated. It might not be supported in future versions. Please use detect() or track() instead.
	virtual void process(const cv::Mat& image) { bool not_supported = false; assert(not_supported); };

	/// Detect landmarks
	virtual void detect(const cv::Mat& image) = 0;
	virtual void detect(const cv::Mat& image, const cv::Rect& bbox) { detect(image); }

	/// Track landmarks
	virtual void track(const cv::Mat& image) { detect(image); }
	virtual void track(const cv::Mat& image, const cv::Rect& bbox) { track(image, bbox); }

	/// Get landmarks (return true if available)
	virtual bool get_landmarks(std::vector<cv::Point2f> & points) { return false; }

	/// Get head pose angles (return true if available)
	virtual bool get_pose_angles(float & pitch, float & yaw, float & roll) { return false; }

	virtual void plot_landmarks(cv::Mat & image, const std::vector<cv::Point2f> & points, const int pt_radius = 3.0, const cv::Scalar color = CV_RGB(0,255,0)) const
	{
		for (size_t i = 0; i < points.size(); ++i)
			cv::circle(image, points[i], pt_radius, color, -1);
	}

	/// Reinitialize (for tracking methods only)
	virtual void reinitialize() {};
};
