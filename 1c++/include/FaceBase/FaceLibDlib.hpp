#pragma once

#include <FaceBase/FaceLib.hpp>
#include <FaceBase/FaceDetector2D.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <vector>

class FaceLibDlib : public FaceLib
{
private:

	mutable cv::Ptr<FaceDetector2D> m_face_detector;
    dlib::shape_predictor m_shape_predictor;
	std::vector<cv::Point2f> m_landmarks;
	bool processed;
public:
	FaceLibDlib() : FaceLib("Dlib"), m_face_detector(), processed(false)
	{
	}

	~FaceLibDlib()
	{
	}

	bool init(cv::Ptr<FaceDetector2D> & face_detector, const char * detection_model_fn = "shape_predictor_68_face_landmarks.dat")
	{
		try
		{
			// Load shape predictor
			dlib::deserialize(detection_model_fn) >> m_shape_predictor;

			// Set active face detector
			m_face_detector = face_detector;

			return true;
		}
		catch(...)
		{
			return false;
		}
	}

	/// Process image
	// This method is deprecated and will no longer be supported. Please use detect() instead.
	void process(const cv::Mat& cv_img)
	{
		processed = false;

		cv::Rect bbox;
		if(m_face_detector && m_face_detector->detect_face(cv_img, bbox))
		{
			detect(cv_img, bbox);
		}

	}

	void detect(const cv::Mat& cv_img)
	{
		processed = false;

		cv::Rect bbox;
		if(m_face_detector && m_face_detector->detect_face(cv_img, bbox))
		{
			detect(cv_img, bbox);
		}

	}

	void detect(const cv::Mat& cv_img, const cv::Rect & bbox)
	{
		processed = false;

		// We need 8bit input images
		if(cv_img.depth() != CV_8U)
			return;

		// Convert opencv rect to dlib rectangle
		dlib::rectangle rect = dlib::rectangle(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);

		dlib::array2d<unsigned char> img_char;
		dlib::array2d<dlib::rgb_pixel> img_rgb;
		dlib::full_object_detection shape;

		switch(cv_img.channels())
		{
		case 1: // graylevel
			// Convet image to dlib image
			dlib::assign_image(img_char, dlib::cv_image<unsigned char>(cv_img));
			// Detect points
			shape = m_shape_predictor(img_char, rect);
			break;

		case 3: // bgr image
			// Convet image to dlib image
			dlib::assign_image(img_rgb, dlib::cv_image<dlib::bgr_pixel>(cv_img));
			// Detect points
			shape = m_shape_predictor(img_rgb, rect);
			break;

		default:
			return;
		}		

		// Do not save points if not enough points are detected (I havent seen that this can happen, but lets be prepared.)
		if(shape.num_parts() != m_shape_predictor.num_parts())
			return;

		// Save points
		m_landmarks.resize(shape.num_parts());
		for(size_t p = 0; p < shape.num_parts(); p++)
		{
			m_landmarks.at(p).x = shape.part(p).x();
			m_landmarks.at(p).y = shape.part(p).y();
		}
			
		processed = true;
	}

	/// Get landmarks (return true if available)
	virtual bool get_landmarks(std::vector<cv::Point2f> & points) 
	{
		if(processed)
			points = m_landmarks;
		else
			points.clear();
		return processed;
	}

	//template<typename I, typename O>
	static void conv_landmarks_68_to_49(const std::vector<cv::Point2f>& pointsIn, std::vector<cv::Point2f>& pointsOut)
	{
		CV_Assert(pointsIn.size() == 68);
		const static int mask[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67};
		pointsOut.clear();
		for (int i = 0; i < 49; ++i)
		{
			pointsOut.push_back(pointsIn[mask[i]]);
		}
	}

};