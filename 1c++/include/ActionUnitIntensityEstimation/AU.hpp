#pragma once

#include <opencv2/core/core.hpp>

class AUIntensityEstimation
{
public:
	inline AUIntensityEstimation()
	{
		m_init_feat = false;
		m_init_mean_std = false;
		m_init_regressor = false;
	}

	inline AUIntensityEstimation(const std::string & feature_model_filename, const std::string & mean_std_filename, const std::string & regressor_filename)
	{
		m_init_feat = false;
		m_init_mean_std = false;
		m_init_regressor = false;
		init(feature_model_filename, mean_std_filename, regressor_filename);
	};

	bool init(const std::string & feature_model_filename, const std::string & mean_std_filename, const std::string & regressor_filename);


	bool estimate(const cv::Mat & image_registered, const std::vector<cv::Point2f> landmarks_registred, cv::Mat & AUintensities) const;
	

	void visualize(cv::Mat & image, const std::vector<cv::Point2f> & landmarks, const cv::Mat & AUintensities, const std::vector<int> & AUIds = std::vector<int>(), const cv::Rect & bbox = cv::Rect(0,0,0,0)) const;

	inline bool is_initialized() const {
		return m_init_feat && m_init_mean_std && m_init_regressor;
	}

	// 1xN matrix (1 row, N columns) of type CV_32SC1 (int) containing the action unit id for each action unit.
	inline cv::Mat get_AUIds() const {
		if(is_initialized())
			return m_AUIds;
		else
			return cv::Mat(); // Return empty matrix if model hasn't been initialized properly.
	}

private:

	bool load_feature_model(const std::string & filename);
	bool load_mean_std(const std::string & filename);
	bool load_regression_model(const std::string & filename);


	bool extract_features(const cv::Mat & image_registered, cv::Mat & features) const;
	bool combine_and_normalize(const std::vector<cv::Point2f> & landmarks, const cv::Mat & lbp_features, cv::Mat & features_combined) const;
	bool regression(const cv::Mat & feature, cv::Mat & prob) const;


	bool lbp(const cv::Mat& image, cv::Mat& feature) const;
	cv::Mat roundn(const cv::Mat& x, const int n) const;
	double roundn(const double x, const int n) const;


	float dist_lm(const cv::Point2f a, const cv::Point2f b) const;
	cv::Point2f center_lm(const std::vector<cv::Point2f> points) const;

	// Feature params
	bool   m_init_feat;
	bool   m_init_mean_std;
	bool   m_init_regressor;

	unsigned int m_lbp_num_blocks_x;
	unsigned int m_lbp_num_blocks_y;
	unsigned int m_lbp_samples;
	unsigned int m_lbp_num;
	unsigned int m_lbp_radius;
	unsigned int m_lbp_neighbors;
	cv::Mat m_lbp_table;
	cv::Mat m_lbp_num_mapping;

	cv::Mat m_mean;
	cv::Mat m_std_inv;

	// Regressor params
	cv::Mat m_AUIds;
	cv::Mat m_rho;
	cv::Mat m_wt;
};