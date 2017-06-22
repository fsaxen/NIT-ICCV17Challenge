#pragma once

#include <FaceBase/FaceRegistration.hpp>

/// Face registration through affine transformation by minimising mean square error between landmarks and a mean shape
class FaceRegistrationAffineMeanShape : public FaceRegistration
{
	std::vector<cv::Point2f> mean_shape;
	cv::Size output_size;

public:

	FaceRegistrationAffineMeanShape() : FaceRegistration("FaceRegistrationAffineMeanShape") {}

	/*!
	 *	\brief Load mean shape model
	 *	\param mean_shape_filename	File to load mean shape from (format: x1 \t y1 \t x2 \t y2 ... )
	 *	\param aligned_img_size		Size of registration output image
	 *	\param eye_dist_frac		Fraction of image that should be covered by eye distance
	 */
	bool init(const char * mean_shape_filename = "mean_face_shape_intraface.dat", const cv::Size & aligned_img_size = cv::Size(180, 200), float eye_dist_frac = 0.5);

	/// Register face, needs landmarks coordinates inside image, returns transformed image and/or landmarks if needed
	bool register_face(const std::vector<cv::Point2f> & in_landmarks, const cv::Mat & in_image, cv::Mat * transformed_image = NULL, std::vector<cv::Point2f> * transformed_landmarks = NULL);

	bool visualize_landmarks(cv::Mat & image, const std::vector<cv::Point2f> & landmark, bool mean_shape = true);
};
