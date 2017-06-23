function ml_param = ml_param()
% Mashine learning parameters for ml.train().
%
% ml_param
%   .type =
%       'lm':   Fit linear model using lmfit (default)
%       'RFc':  Random Forest classification
%       'RFr':  Random Forest regression
%       'SVM', 'SVR':   Support Vector Machine or Regression. Please
%                       specify in ml_param.svm_param.
%       'EasyEnsemble': Easy Ensemble classification.
%       'Ensemble'    : Splits the dataset into several chunks and trains a
%                       new model for each set. An aggregation model
%                       combines all trained model to provide a single
%                       output.
%   
%   .num_cpu_cores =
%       Number of cpu cores to use for training. Only used if multiple
%       predictors available for training. (default 1)
%
%   .svm_param =
%       Structure of svm parameter used when ml_param.type = 'SVM' or 'SVR'.
%       See svm.create_svm_param.m for details.
%
%   .ee_param =
%       Easy Ensemble parameter, used when ml_param.type = 'EasyEnsemble'.
%       See EasyEnsemble.create_EasyEnsemble_param.m for details.
%
%   .ensemble_param =
%       Nested ml_param to train an ensemble of SVM, SVR, or RFc or
%       whatsoever. Used when ml_param.type = 'Ensemble'. See
%       ml.ml_param.m for details ;) 
%
%   .ensemble_num_models =
%       Number of models in the ensemble. Used, when ml_param.type =
%       'Ensemble'. (default 4)
%
%   .num_samples =
%       Number of samples used for training. If
%       ml_param.type ~= 'Ensemble', the default is to use all samples.
%       With ml_param.type = 'Ensemble', each ensemble model will get
%       independently sampled data for training. By default its set to the
%       number of samples in the dataset times 2 / ensemble_num_models.
%       Thus, all sampled subsets together contain twice the number of
%       samples compared to the original dataset.
%
%   .redistribute_param =
%       Redistribution parameter to change the skew of the dataset. Please
%       see libDataset.redistribute_param.m for parameter details.
%
%   .param_search =
%       Parameters for running a model parameter grid search.
%       This is a struct with the following member variables:
%       .params 
%           Cell array of string and array pairs. Each string names a
%           ml_param member variable (e.g. 'svm_param.C') that should be
%           varied. The following array contains the values that should be
%           tested.
%           Example:
%                param_search.params = { ...
%                        'svm_param.C', 2.^(-5:10:25),...
%                        'svm_param.gamma', 2.^(-15:5:-5) };
%       .enable (optional)
%           Boolean. Set false to disable search. Default: true
%       .max_num_training_samples (optional)
%           Maximum number of samples to be used for training during
%           parameter evaluation. Default: 1000
%       .max_num_testing_samples (optional)
%           Maximum number of samples to be used for testing during
%           parameter evaluation. Default: 10000
%       .split_sampling (optional)
%           Split criterion used to create training and test set. See
%           libDataset.split_param.m for parameter detail.
%           Default: 'random_subjects' if subject field is available in
%           dataset, 'stratified' if predictor is discrete, 'random'
%           otherwise
%       .filename_prefix (optional)
%           If this is defined (and not empty), parameter validation
%           results will be visualized in plots that are saved as PNG image
%           files. The content of this variable (string) will be used as
%           prefix.

ml_param = struct;

end
