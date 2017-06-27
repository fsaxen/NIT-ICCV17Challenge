// Authors: Frerk Saxen and Philipp Werner (Frerk.Saxen@ovgu.de, Philipp.Werner@ovgu.de)
// License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

#include <ActionUnitIntensityEstimation/AU.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <cstdio>
#include <iostream>
#define PI 3.14159265358979323846


bool AUIntensityEstimation::init(const std::string & feature_model_filename, const std::string & mean_std_filename, const std::string & regressor_filename)
{
	return load_feature_model(feature_model_filename) && load_mean_std(mean_std_filename) && load_regression_model(regressor_filename);
}

bool AUIntensityEstimation::estimate(const cv::Mat & image_registered, const std::vector<cv::Point2f> landmarks_registred, cv::Mat & AUintensities) const
{
	if(!is_initialized())
		return false;

	cv::Mat feat, feat_combined;

	if(!extract_features(image_registered, feat))
		return false;

	if(!combine_and_normalize(landmarks_registred, feat, feat_combined))
		return false;

	if(!regression(feat_combined, AUintensities))
		return false;

	return true;
}

// Colors from http://tools.medialab.sciences-po.fr/iwanthue/
//const cv::Scalar colors[12] = {
//	CV_RGB(214,104,48), 
//	CV_RGB(127,109,205),
//	CV_RGB(98,178,63),
//	CV_RGB(75,147,150),
//	CV_RGB(199,87,147),
//	CV_RGB(172,152,56),
//	CV_RGB(121,76,43),
//	CV_RGB(103,74,113),
//	CV_RGB(204,82,210),
//	CV_RGB(81,133,79),
//	CV_RGB(199,78,86),
//	CV_RGB(131,149,200) 
//};


void AUIntensityEstimation::visualize(cv::Mat & image, const std::vector<cv::Point2f> & landmarks, const cv::Mat & AUintensities, const std::vector<int> & AUIds, const cv::Rect & bbox) const
{
	cv::Mat image_int = image;
	std::vector<cv::Point2f> landmarks_int = landmarks;

	if(bbox.width != 0 && bbox.height != 0)
	{
		image_int = image_int(bbox);
		for(int i = 0; i < landmarks_int.size(); i++)
		{
			landmarks_int.at(i).x -= bbox.x;
			landmarks_int.at(i).y -= bbox.y;
		}
	}

	//cv::Mat r, g, b;
	cv::Mat h = cv::Mat::zeros(image_int.size(), CV_32FC1);
	cv::Mat s = cv::Mat::zeros(image_int.size(), CV_32FC1);
	cv::Mat v = cv::Mat::ones(image_int.size(), CV_32FC1);

	cv::Point2f centerLeft, centerRight, center;
	float radiusLeft, radiusRight, radius;
	std::vector<cv::Point2f> points;
	cv::Mat pointsMat;

	std::vector<int> AUsToPlot = AUIds;
	if(AUIds.empty())
		for(int i = 0; i < m_AUIds.cols; i++)
			AUsToPlot.push_back(m_AUIds.at<int>(i));

	for(int i = 0; i < AUsToPlot.size(); i++)
	{
		// Get action unit to plot
		int au = AUsToPlot.at(i);
		// Find index of current action unit in AUintensities matrix
		int auIdx = 0;
		for(; auIdx < AUintensities.cols; auIdx++)
			if(m_AUIds.at<int>(auIdx) == au)
				break;
		// If index does not exist because the current model does not provide this AU, just scip it.
		if(m_AUIds.at<int>(auIdx) != au)
			continue;

		float auIntensity = AUintensities.at<float>(auIdx);

		switch(au)
		{
		case 1: // Inner brow raiser
			// Plot a circle on inner brows
			// left
			centerLeft = landmarks_int[3];
			radius = auIntensity * image_int.cols / 50.0;
			//radiusLeft = dist_lm(landmarks_int[4], landmarks_int[3]);
			cv::circle(h, centerLeft, radius, 30, -1);
			cv::circle(s, centerLeft, radius, auIntensity * 0.2, -1);
			// right
			centerRight = landmarks_int[6];
			//radiusRight = dist_lm(landmarks_int[5], landmarks_int[6]);
			cv::circle(h, centerRight, radius, 30, -1);
			cv::circle(s, centerRight, radius, auIntensity * 0.2, -1);
			break;

		case 2: // Outer brow raiser
			// Plot a circle on outer brows
			// left
			centerLeft = landmarks_int[1];
			radius = auIntensity * image_int.cols / 50.0;
			//radiusLeft = dist_lm(landmarks_int[1], landmarks_int[2]);
			cv::circle(h, centerLeft, radius, 30, -1);
			cv::circle(s, centerLeft, radius, auIntensity * 0.2, -1);
			// right
			centerRight = landmarks_int[8];
			//radiusRight = dist_lm(landmarks_int[8], landmarks_int[7]);
			cv::circle(h, centerRight, radius, 30, -1);
			cv::circle(s, centerRight, radius, auIntensity * 0.2, -1);
			break;

		case 4: // Brow lowerer
			// Plot a circle between the eyes
			points.clear();
			points.push_back(landmarks_int[4]);
			points.push_back(landmarks_int[5]);
			//points.push_back(landmarks_int[10]);
			center = center_lm(points); 
			//radius = dist_lm(center, landmarks_int[4]);
			radius = auIntensity * image_int.cols / 50.0;
			cv::circle(h, center, radius, 0, -1);
			cv::circle(s, center, radius, auIntensity * 0.2, -1);
			break;

		case 6: // cheek raiser
			// Plot a circle on cheek
			// left
			points.clear();
			points.push_back(landmarks_int[0]);
			points.push_back(landmarks_int[31]);
			centerLeft = center_lm(points);
			radius = auIntensity * image_int.cols / 50.0;
			//radiusLeft = dist_lm(landmarks_int[19], landmarks_int[24]);
			cv::circle(h, centerLeft, radius, 300, -1);
			cv::circle(s, centerLeft, radius, auIntensity * 0.2, -1);
			// right
			points.clear();
			points.push_back(landmarks_int[9]);
			points.push_back(landmarks_int[37]);
			centerRight = center_lm(points);
			//radiusRight = dist_lm(landmarks_int[28], landmarks_int[29]);
			cv::circle(h, centerRight, radius, 300, -1);
			cv::circle(s, centerRight, radius, auIntensity * 0.2, -1);
			break;

		//case 9: // nose wrinkle
		//	// Plot a circle between the eyes
		//	center = landmarks_int[11]; 
		//	radius = dist_lm(landmarks_int[12], landmarks_int[11]);
		//	cv::circle(h, center, radius, 25, -1);
		//	cv::circle(s, center, radius, auIntensity * 0.2, -1);
		//	break;

		case 12:
			// Plot a circle on left and right mouth corner
			// left
			centerLeft = landmarks_int[31];
			radiusLeft = dist_lm(landmarks_int[31], landmarks_int[42]);
			cv::circle(h, centerLeft, radiusLeft, 120, -1);
			cv::circle(s, centerLeft, radiusLeft, auIntensity * 0.20, -1);
			// right
			centerRight = landmarks_int[37];
			radiusRight = dist_lm(landmarks_int[37], landmarks_int[38]);
			cv::circle(h, centerRight, radiusRight, 120, -1);
			cv::circle(s, centerRight, radiusRight, auIntensity * 0.20, -1);
			
			break;

		case 25:
			// Plot a polygon in the opened mouth
			pointsMat = cv::Mat(0,0,CV_32FC1);
			pointsMat.push_back(landmarks_int[31]);
			pointsMat.push_back(landmarks_int[43]);
			pointsMat.push_back(landmarks_int[44]);
			pointsMat.push_back(landmarks_int[45]);
			pointsMat.push_back(landmarks_int[37]);
			pointsMat.push_back(landmarks_int[46]);
			pointsMat.push_back(landmarks_int[47]);
			pointsMat.push_back(landmarks_int[48]);
			pointsMat.push_back(landmarks_int[31]);
			pointsMat.convertTo(pointsMat, CV_32SC1);

			cv::fillConvexPoly(h, pointsMat, 240, CV_AA);
			cv::fillConvexPoly(s, pointsMat, auIntensity * 0.10, CV_AA);
		}
	}

	cv::Mat channels[] = { h, s, v };
	cv::Mat hsv, bgr;
	cv::merge(channels, 3, hsv);
	cv::cvtColor(hsv, bgr, CV_HSV2BGR);
	

	image_int.convertTo(image_int, CV_32FC3);
	image_int = image_int.mul(bgr);
	image_int.convertTo(image_int, CV_8UC3);

	if(bbox.width > 0 && bbox.height > 0)
		image_int.copyTo(image(bbox));
	else
		image_int.copyTo(image);

}

float AUIntensityEstimation::dist_lm(const cv::Point2f a, const cv::Point2f b) const
{
	cv::Point2f d = a - b;
	return std::sqrt(d.x * d.x + d.y * d.y);
}

cv::Point2f AUIntensityEstimation::center_lm(const std::vector<cv::Point2f> points) const
{
	cv::Point2f center = cv::Point2f(0,0);
	for(int i = 0; i < points.size(); i++)
		center += points[i];
	center.x = center.x / points.size();
	center.y = center.y / points.size();
	return center;
}

bool AUIntensityEstimation::load_feature_model(const std::string & filename)
{
	m_init_feat = false;

	FILE* pFile = fopen(filename.c_str(),"r");

	if (pFile==NULL)
		return false;

	fscanf(pFile, "%i %i %i %i", &m_lbp_num_blocks_x, &m_lbp_num_blocks_y, &m_lbp_neighbors, &m_lbp_radius);

	fscanf(pFile, "%i %i", &m_lbp_samples, &m_lbp_num);

	m_lbp_table = cv::Mat(1, 256, CV_8UC1);
	m_lbp_num_mapping = cv::Mat(1, 256, CV_8UC1);

	for(int i = 0; i < 256; i++)
	{
		int v;
		fscanf(pFile, "%i", &v);
		m_lbp_table.at<uchar>(i) = static_cast<uchar>(v);
	}

	for(int i = 0; i < 256; i++)
	{
		int v;
		fscanf(pFile, "%i", &v);
		m_lbp_num_mapping.at<uchar>(i) = static_cast<uchar>(v);
	}

	fclose(pFile);

	m_init_feat = true;
	return true;}

bool AUIntensityEstimation::load_mean_std(const std::string & filename)
{
	m_init_mean_std = false;

	FILE* pFile = fopen(filename.c_str(),"r");

	if (pFile==NULL)
		return false;

	int size_mean, size_std;
	fscanf(pFile, "%i", &size_mean);

	m_mean = cv::Mat(size_mean, 1, CV_32FC1);

	float* p_mean = m_mean.ptr<float>(0);
	for(int i = 0; i < size_mean; i++)
	{
		fscanf(pFile, "%f", p_mean++);
	}

	fscanf(pFile, "%i", &size_std);
	m_std_inv = cv::Mat(size_std, 1, CV_32FC1);

	float* p_std_inv = m_std_inv.ptr<float>(0);
	float v;
	for(int i = 0; i < size_mean; i++)
	{
		fscanf(pFile, "%f", &v);
		p_std_inv[i] = 1.0 / v;
	}

	fclose(pFile);

	m_init_mean_std = true;
	return true;
}

bool AUIntensityEstimation::load_regression_model(const std::string & filename)
{
	m_init_regressor = false;

	FILE* pFile = fopen(filename.c_str(),"r");

	if (pFile==NULL)
		return false;

	int num_AUs;
	fscanf(pFile, "%i", &num_AUs);

	m_AUIds = cv::Mat(1, num_AUs, CV_32SC1);
	int* p_id = m_AUIds.ptr<int>(0);
	for(int au = 0; au < num_AUs; au++)
		fscanf(pFile, "%i", p_id++);
	
	m_rho = cv::Mat(1, num_AUs, CV_32FC1);
	float* p_r = m_rho.ptr<float>(0);
	for(int au = 0; au < num_AUs; au++)
		fscanf(pFile, "%f", p_r++);

	int num_feat;
	fscanf(pFile, "%i", &num_feat);

	m_wt = cv::Mat(num_AUs, num_feat, CV_32FC1);

	float* p_wt = m_wt.ptr<float>(0);
	for(int i = 0; i < num_feat * num_AUs; i++)
	{
		fscanf(pFile, "%f", p_wt++);
	}

	m_wt = m_wt.t();

	fclose(pFile);

	m_init_regressor = true;
	return true;
}



bool AUIntensityEstimation::extract_features(const cv::Mat & image_registered, cv::Mat & features) const
{
	if(!m_init_feat)
		return false;

	// Release feature data
	features = cv::Mat(0,0,CV_64FC1);

	// Convert image to grayscale and divide by 256.0
	cv::Mat im_gray;
	
	if(image_registered.channels() == 3)
	{
		cv::cvtColor(image_registered, im_gray, CV_BGR2GRAY);
		im_gray.convertTo(im_gray, CV_64FC1);
	}
	else
		image_registered.convertTo(im_gray, CV_64FC1);

	im_gray /= 256.0;

	if(im_gray.cols % m_lbp_num_blocks_x != 0 || im_gray.rows % m_lbp_num_blocks_y != 0) {
		std::cout << "Feature extraction error: Image size must be a multiple of feature_param lbp_num_blocks" << std::endl;
		return false;
	}

	// Calculate image step
	unsigned int step_x = im_gray.cols / m_lbp_num_blocks_x;
	unsigned int step_y = im_gray.rows / m_lbp_num_blocks_y;

	cv::Mat im_block, feat_block;
	cv::Mat im_show;
	for(unsigned int x = 0; x < m_lbp_num_blocks_x; x++)
	{
		for(unsigned int y = 0; y < m_lbp_num_blocks_y; y++)
		{
			im_block = im_gray(cv::Rect(x * step_x, y * step_y, step_x, step_y));
			lbp(im_block, feat_block);
			features.push_back(feat_block);
		}
	}

	return true;
}

bool AUIntensityEstimation::combine_and_normalize(const std::vector<cv::Point2f> & landmarks, const cv::Mat & lbp_features, cv::Mat & features_combined) const
{
	if(!m_init_mean_std)
		return false;

	// Allocate memory for landmarks
	features_combined = cv::Mat(landmarks.size() * 2, 1, CV_32FC1);
	
	// Append landmarks
	float* p = features_combined.ptr<float>(0);
	for(int i = 0, j = 0; i < landmarks.size(); i++, j+=2) {
		p[j] = landmarks[i].x;
		p[j+1] = landmarks[i].y;
	}

	// Append features
	features_combined.push_back(lbp_features);

	features_combined -= m_mean;
	features_combined = features_combined.mul(m_std_inv);

	// Done!
	return true;
}

bool AUIntensityEstimation::regression(const cv::Mat & feature, cv::Mat & prob) const
{
	if(!m_init_regressor)
		return false;

	prob = feature.t() * m_wt - m_rho;
	//prob = cv::max(prob, 0.0);

	return true;
}


bool AUIntensityEstimation::lbp(const cv::Mat& image, cv::Mat& feature) const
{
	double neighbors = static_cast<double>(m_lbp_neighbors);
	double radius = static_cast<double>(m_lbp_radius);

	// angle step
	double a = 2.0 * PI / neighbors;

	// Calculate spoint values
	cv::Mat spoints(m_lbp_neighbors, 2, CV_64FC1);
	for(unsigned int i = 0; i < m_lbp_neighbors; i++)
	{
		spoints.at<double>(i, 0) = -radius * sin(i * a);
		spoints.at<double>(i, 1) = radius * cos(i * a);
	}

	// Get dimensions of input image
	int ysize = image.rows;
	int xsize = image.cols;

	// Calculate minimum and maximum indices
	double miny, maxy, minx, maxx;
	cv::minMaxLoc(spoints.col(0), &miny, &maxy);
	cv::minMaxLoc(spoints.col(1), &minx, &maxx);

	// Block size, each LBP code is computed within a block of size bsizey*bsizex
	int bsizey = static_cast<int>(std::ceil(std::max(maxy,0.0)) - std::floor(std::min(miny,0.0)) + 1);
	int bsizex = static_cast<int>(std::ceil(std::max(maxx,0.0)) - std::floor(std::min(minx,0.0)) + 1);

	// Coordinates of origin (0,0) in the block
	int origy = -static_cast<int>(floor(std::min(miny,0.0)));
	int origx = -static_cast<int>(floor(std::min(minx,0.0)));

	// Minimum allowed size for the input image depends
	// on the radius of the used LBP operator.
	if(xsize < bsizex || ysize < bsizey)
	{
		printf("Too small input image. Should be at least (2*radius+1) x (2*radius+1)");
		return false;
	}

	// Calculate dx and dy;
	int dx = xsize - bsizex + 1;
	int dy = ysize - bsizey + 1;

	// Fill the center pixel matrix C.
	cv::Mat C = image(cv::Rect(origx,origy,dx,dy)).clone(); // Why do I need to copy the data here?

	double bins = pow(2.0,neighbors);

	// Initialize the result matrix with zeros.
	cv::Mat result=cv::Mat::zeros(dx,dy,CV_8UC1);

	//Compute the LBP code image
	cv::Mat N, D;
	for(int i = 0; i < neighbors; i++)
	{
		double y = spoints.at<double>(i,0)+origy;
		double x = spoints.at<double>(i,1)+origx;
		
		// Calculate floors, ceils and rounds for the x and y.
		int fy = static_cast<int>(floor(y)), cy = static_cast<int>(ceil(y)), ry = cvRound(y);
		int fx = static_cast<int>(floor(x)), cx = static_cast<int>(ceil(x)), rx = cvRound(x);
		
		// Check if interpolation is needed.
		if((abs(x - rx) < 1e-6) && (abs(y - ry) < 1e-6))
		{
			// Interpolation is not needed, use original datatypes
			N = image(cv::Rect(rx,ry,dx,dy)).clone(); // And why do I need to copy the data here?
			D = N >= C; 
		}
		else
		{
			// Interpolation needed, use double type images 
			double ty = y - fy;
			double tx = x - fx;

			// Calculate the interpolation weights.
			double w1 = roundn((1 - tx) * (1 - ty),-6);
			double w2 = roundn(tx * (1 - ty),-6);
			double w3 = roundn((1 - tx) * ty,-6) ;
			double w4 = roundn(1 - w1 - w2 - w3, -6);
            
			// Compute interpolated pixel values
			N = w1 * image(cv::Rect(fx,fy,dx,dy))  +  w2 * image(cv::Rect(cx,fy,dx,dy))  +  w3 * image(cv::Rect(fx,cy,dx,dy))  +  w4 * image(cv::Rect(cx,cy,dx,dy));
			N = roundn(N,-4);
			D = N >= C; 
		}

		// Update the result matrix.
		double v = pow(2.0, i);
		D = D / 255.0;
		//D.convertTo(D, CV_32SC1); // Convert D to 32 bit integer (not necessary if neighbors = 8 and radius = 1)
		result = result + D * v;
	}

	// Apply mapping
	cv::LUT(result, m_lbp_table, result);

    // Return with LBP histogram.
	const int channels[] = {0};
	const int histSize[] = {static_cast<int>(m_lbp_num)};
	const float range[] = {0, static_cast<float>(m_lbp_num)};
	const float* ranges[] = {range};
	cv::calcHist(&result, 1, channels, cv::Mat(), feature, 1, histSize, ranges, true, false);

	// Done !	
	return true;
}

cv::Mat AUIntensityEstimation::roundn(const cv::Mat& x, const int n) const
{
	// Accept only double matrices
	CV_Assert(x.type() == CV_64FC1);

	cv::Mat r = cv::Mat(x.size(),x.type());

	int nRows = x.rows;
    int nCols = x.cols;

	if (x.isContinuous() && r.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    const double* px;
	double* pr;
    for( i = 0; i < nRows; ++i)
    {
        px = x.ptr<double>(i);
		pr = r.ptr<double>(i);
        for ( j = 0; j < nCols; ++j)
        {
            pr[j] = roundn(px[j],n);
        }
    }

    return r;
}

double AUIntensityEstimation::roundn(const double x, const int n) const
{
	double p;
	if(n < 0)
	{
		p = pow(10.0,-n);
		return(static_cast<double>(cvRound(p * x)) / p);
	}
	else if(n > 0)
	{
		p = pow(10.0,n);
		return(p * cvRound(x / p));
	}
	else
		return(cvRound(x));
}
