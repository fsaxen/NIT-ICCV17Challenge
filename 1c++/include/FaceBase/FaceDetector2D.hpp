#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

/// Base class for 2D face detector
class FaceDetector2D
{
public:
	FaceDetector2D()							{}
	virtual ~FaceDetector2D()					{ release(); }

	virtual bool load(const char * model_fn = 0)	{ return false; }
	virtual void release()						{}

	virtual bool detect_face(const cv::Mat & img, cv::Rect & face_bbox) const
	{ 
		std::vector<cv::Rect> face_bboxs;
		if(!detect_faces(img, face_bboxs))
			return false;

		face_bbox = face_bboxs.at(0);
		return true;
	}

	virtual bool detect_faces(const cv::Mat & img, std::vector<cv::Rect> & face_bboxs) const = 0;
};
