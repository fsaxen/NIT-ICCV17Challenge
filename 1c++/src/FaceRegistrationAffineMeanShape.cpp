#include <FaceBase/FaceRegistrationAffineMeanShape.hpp>

#include <iostream>
#include <fstream>
#include <opencv2/video/video.hpp>



/* Estimates coefficients of affine transformation
 * which approximatly maps (xi,yi) to (ui,vi), (i=1,...,n):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \       /u0\
 * | x1 y1  1  0  0  0 | /c00\ |u1|
 * | xn yn  1  0  0  0 | |c01| |un|
 * |    .   .   .      | |c02| | .|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * |    .   .   .      | \c12/ | .|
 * \  0  0  0 xn yn  1 /       \vn/
 *
 * where:
 *   cij - matrix coefficients
 */
cv::Mat estimateAffineTransform(const std::vector<cv::Point2f> & src, const std::vector<cv::Point2f> & dst)
{
	CV_Assert(src.size() == dst.size());
    cv::Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.data);
	cv::Mat1d A(2*(int)src.size(), 6), B(2*(int)src.size(), 1);
	double * a;

    for( int i = 0; i < src.size(); i++ )
    {
		const cv::Point2f & ps = src[i];
		const cv::Point2f & pd = dst[i];

        B(2*i) = pd.x;
		a = A.ptr<double>(2*i);
        a[0] = ps.x;
        a[1] = ps.y;
        a[2] = 1;
        a[3] = a[4] = a[5] = 0;

        B(2*i+1) = pd.y;
		a = A.ptr<double>(2*i+1);
        a[0] = a[1] = a[2] = 0;
        a[3] = ps.x;
        a[4] = ps.y;
        a[5] = 1;
    }

	cv::solve( A, B, X, cv::DECOMP_SVD );
    return M;
}



bool FaceRegistrationAffineMeanShape::init(const char * mean_shape_filename, const cv::Size & aligned_img_size, float eye_dist_frac)
{
	output_size = aligned_img_size;
	float x_scale = aligned_img_size.width * eye_dist_frac;
	float x_offset = 0.5f * aligned_img_size.width;
	float y_scale = x_scale;
	float y_offset = 0.5f * aligned_img_size.height;

	std::ifstream file(mean_shape_filename);
	mean_shape.clear();
	float x, y;
	while (file >> x >> y)
		mean_shape.push_back(cv::Point2f(x * x_scale + x_offset, y * y_scale + y_offset));

	return !mean_shape.empty();
}

bool FaceRegistrationAffineMeanShape::register_face(const std::vector<cv::Point2f> & in_landmarks, const cv::Mat & in_image, cv::Mat * transformed_image, std::vector<cv::Point2f> * transformed_landmarks)
{
	if (mean_shape.size() != in_landmarks.size())
		return false;

	bool fullAffine = false;
	cv::Mat affine_transform = estimateAffineTransform(in_landmarks, mean_shape);
	//cv::Mat affine_transform = cv::estimateRigidTransform(in_landmarks, mean_shape, fullAffine);
	if (transformed_image)
		cv::warpAffine(in_image, *transformed_image, affine_transform, output_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

	if (transformed_landmarks)
		cv::transform(in_landmarks, *transformed_landmarks, affine_transform);


	return true;
}

bool FaceRegistrationAffineMeanShape::visualize_landmarks(cv::Mat & image, const std::vector<cv::Point2f> & landmarks, bool draw_mean_shape)
{
	for (size_t i = 0; i < landmarks.size(); ++i)
		cv::circle(image, landmarks[i], 1, CV_RGB(0,255,0),-1);

	if (draw_mean_shape)
		for (size_t i = 0; i < mean_shape.size(); ++i)
			cv::circle(image, mean_shape[i], 1, CV_RGB(255,0,255),-1);

	return true;
}

