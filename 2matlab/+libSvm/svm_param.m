function param = svm_param()

% svm_param
%   .library =
%       'libsvm': Normal svm library which is pretty slow for big datasets (default).
%       'liblinear' : Fast linear support vector machine.
%
%   .predict_fast =
%       'true': Use fast matlab code for prediction (default).
%       'false': Use original libsvm or liblinear code.
%
%   .type =
%       'SVMm': Support Vector Machine - Multiclass  (default)
%       'SVMb': Support Vector Machine - Binary (two-class); result is a
%               linear function of decision value
%       'SVR':  Support Vector Regression
%
%   .kernel =
%       'linear':        linear kernel (default) -  x*sv'
%       'rbf':           gauss kernel            -  exp(-gamma * (x-sv).2)
%       'polynomial':    polynomial kernel       -  (gamma * x * sv' + coef0).^degree
%       'sigmoid':       sigmoid kernel          -  tanh(gamma * x * sv' + coef0) 
%       'intersection':  intersection kernel     -  sum(min(x,sv))
%
%   .C =
%       The ordinary C parameter of the svm (default 1.0). This will be set
%       if param_search.enable = true.
%
%   .epsilon =
%       The epsilon value for support vector regression (SVR) (default 0.1)
%
%   .gamma =
%       The gamma value for the rbf, polynomial, and sigmoid kernel
%       (default 1.0/num_features).
%
%   .degree =
%       Degree for polynomial kernel (default 3.0).
%
%   .coef0 =
%       Coefficient in polynomial and sigmoid kernel (default 0).
%
%   .param_search
%       .enable =
%           false:  No parameter seach (default)
%           true:   Parameter seach will start in train.m.
%                   After the training svm_param.C and other kernel
%                   parameters will be set to the optimal value.
%       
%       .C_range = 
%           Grid search range for C values (default 2 .^ (-5:2:15) ).
%
%       .epsilon_range =
%           Grid search range for epsilon values (default 0.1 = no grid
%           search).
%
%       .gamma_range =
%           Grid search range for gamma values (default 15 .^ (-6:2) ).
%
%       .degree_range =
%           Grid search range for degree values (default 3.0 = no grid
%           search).
%
%       .coef0_range =
%           Grid search range for coef0 values (default 0 = no grid
%           search).
%                   
%       .max_num_training_samples =
%           Maximum number of training samples in the parameter search.
%           For parameter search 1/3 of the dataset is used for training,
%           2/3 is used for validation. But if the dataset becomes too
%           large, you might want to decrease the training set to a maximum
%           value. If this field is not set (or its value smaller than 0)
%           max 1000 samples are used for training (default 1000).
%
%       .performance_measure =
%           In order to compare the svms trained with different parameters,
%           you need to provide a performance measure. See library
%           +performance for a choice of performance measures.
%
%       .print =
%           true:  Print parameter search to file. For linear or
%                  intersection kernels, a 1D plot is saved to file
%                  param_sarch.filename. For rbf kernel a 2D plot is saved.
%           false: Do not print parameter search to file (default).
%
%       .filename =
%           Relative filename to root directory if param_search.print =
%           true.
%
%   .reduce_support_vectors =
%       true:  Calculate weight vector and discard support vectors to save
%              memory and time (for linear kernel and libsvm only).
%       false: Save all support vectors in the trained model (default).
%
%   .cache =
%       Cache size parameter (in MB). If not set, it will be computed. Only
%       valid for libsvm.
%
%   .verbose =
%       true: Print intermediate svm results on command (for debug perpose)
%       false: Train the model quitely (default)
%
%   .fit_correction_function
%       .enable =
%           true:  Fit output class label range of binary or multiclass to
%                  the input class label range. Especially needed if the
%                  input class labels are not integer values but floats.
%           false: No output correction (default)
%
%       .degree =
%           polynomial degree of correction
%
%       .visualize =
%           true:   Visualize the training
%           false:  Don't show training results (default)
%       
%
param = struct;

end