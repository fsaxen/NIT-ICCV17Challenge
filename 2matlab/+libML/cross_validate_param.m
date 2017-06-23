function param = cross_validate_param()
% cross_validate_param
%
% param
%   .ml_param = 
%       Mashine learning parameters for ml.train(). Please see
%       libML.ml_param.m for details.
%
%   .k =
%       Number of folds for the k-fold cross-validation. (default 10)
%
%   .split_method =
%       Any method that offers split_param.split_method. See
%       libDataset.split_param for details. (default 'random')
%
%   .num_folds =
%       Number of executed folds. The cross validation will stop after
%       num_folds folds and not proceed all k possible folds. On default
%       this value will be set to k and is mainly used for debug purpose.
%       (default cross_validate_param.k)
%
%   .performance_measure =
%       A string of one performance measure or a cell array of multiple
%       performance measures. See list of implemented performance 
%       measurements in libPerformance.value(). (default 'mse')
%


    param = struct();
end