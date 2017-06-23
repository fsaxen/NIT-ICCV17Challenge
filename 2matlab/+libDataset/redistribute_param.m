function param = redistribute_param()

% redistribute_param
%   .type =
%       'min' :      Flat output distribution equal to the minimum input
%                    predictor distribution (default).
%       'max' :      Flat output distribution equal to the maximum input
%                    predictor distribution.
%       'mean' :     Flat output distribution equal to the mean input
%                    predictor distribution.
%       'median' :   Flat output distribution equal to the median input
%                    predictor distribution.
%       'ith_order': Flat output distribution equal to the ith input
%                    predictor distribution sorted in ascending order from
%                    least frequent to most frequenct class.
%       'damping':   Curved output distribution whereas the predictor
%                    distribution is calculated for each predictor value
%                    independently with the damping parameters k, beta, and
%                    alpha.
%
%   .i =
%       i sorted predictor value for redistribute_param.type = 'ith_order'.
%       Of course, i must be in the range 1 <= i <= num_classes.
%
%   .downsampling_type =
%       'none':     No downsampling.
%       'random':   Random downsampling (default)
%       'kmeans':   Downsample based on nearest neighbor in feature space.
%
%   .upsampling_type =
%       'none':     No upsampling.
%       'random':   Random upsampling (default)
%       'smote':  	Upsample based on smote (Synthetic Minority
%                   Over-sampling Technique).
%
%   .factor =
%       In order to manually increase or reduce the number of output
%       samples, you can specify a factor that is multiplied to the desired
%       number of output samples. E.g., if you use redistribute_param.type
%       = 'min' but this still results in too many samples, choose a factor
%       < 1. If you want to level the distribution to the minimum but
%       expect more samples in each predictor, choose factor > 1. (default
%       1)
%
%   .damping
%       .alpha = 
%       .beta = 
%       .k =
%
%   .visualize =
%       false:  Don't plot input and output distribution (default)
%       true :  Plot input and output distribution.
%
%   .figure =
%       Figure id to plot the distributions. By default a new figure is
%       opened. Only used if visualize = true.

param = struct();

end