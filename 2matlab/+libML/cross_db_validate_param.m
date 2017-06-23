function param = cross_db_validate_param()
% Parameter structure for the cross_db_validate function
%
% param
%   .ml_param = 
%       Mashine learning parameters for ml.train(). Please see
%       libML.ml_param.m for details.
%
%   .load_from_file =
%       true : Load datasets from file. The inputs of cross_db_validate
%              must then be cells containing the filename to the .mat file
%              of the dataset and its variable name in matlab.
%              cross_db_validate() will then load the dataset from file
%              when necessary to safe memory.
%       false: Use the datasets directly passed to the function. This will
%              use more memory but is generally faster (default).
%
%   .performance_measure =
%       Any performance measurement. See list of implemented performance
%       measurements in libPerformance.value().
%
    param = struct();
end