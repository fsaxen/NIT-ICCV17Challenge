function [ svm_param, param_matrix, perf_matrix, traintime_matrix ] = param_search( data, svm_param )
% Searches for optimal svm parameter.
% dataset : Struct element containing (at least):
%   .x = All features of the entire dataset. Each row represents the
%        feature of one sample.
%
%   .y = All labels/predictors in the entire dataset corresponding to .x
%
%   .subject = Subject number corresponding to each sample. Only used, if
%              svm_param.param_search.split_param needs the subject field.
%
%   .sample_idx = Index of samples in x and y, that are allowed to use for
%                 the parameter search procedure.
%
%   .predictor_idx = Index of the current predictor. dataset.y might have
%                    multiple predictors for each sample, but the svm can
%                    only train one predictor at a time.
%

    if isfield(svm_param.param_search, 'max_num_training_samples')
        max_num_training_samples = svm_param.param_search.max_num_training_samples;
    else
        max_num_training_samples = 1000; %1000 samples
    end
    

    % avoid recursive call of this function in model_train()
    param_search_enable__restore = svm_param.param_search.enable;
    svm_param.param_search.enable = false;
    
    % do not visualize training during parameter search
    if isfield(svm_param.param_search, 'visualize')
        visualize__restore = svm_param.visualize;
        svm_param.visualize = false;
    end
    
    % C range
    if isfield(svm_param.param_search, 'C_range')
        C_range = svm_param.param_search.C_range(:);
    else
        C_range = 2 .^ (-5:2:15);
    end
    
    % epsilon range
    if ~strcmp(svm_param.type, 'SVR')
        epsilon_range = 0.1;
    elseif isfield(svm_param.param_search, 'epsilon_range')
        epsilon_range = svm_param.param_search.epsilon_range;
    else
        epsilon_range = 0.1;
    end

    % gamma range
    if strcmp(svm_param.kernel, 'linear') || strcmp(svm_param.kernel, 'intersecton')
        gamma_range = 1;
    elseif isfield(svm_param.param_search, 'gamma_range')
        gamma_range = svm_param.param_search.gamma_range;
    elseif strcmp(svm_param.kernel, 'rbf') 
        gamma_range = 15 .^ (-6:2);
    else
        gamma_range = 1;
    end
    
    % degree range
    if strcmp(svm_param.kernel, 'polynomial') && isfield(svm_param.param_search, 'degree_range')
        degree_range = svm_param.param_search.degree_range;
    else
        degree_range = 3.0;
    end
    
    % coef0 range
    if strcmp(svm_param.kernel, 'polynomial') || strcmp(svm_param.kernel, 'sigmoid')
        if isfield(svm_param.param_search, 'coef0_range')
            coef0_range = svm_param.param_search.coef0_range;
        else
            coef0_range = 0;
        end
    else
        coef0_range = 0;
    end


    trials = 3;
    n_samples = length(data.sample_idx);
    if n_samples < 3 * max_num_training_samples || max_num_training_samples <= 0
        max_num_training_samples = floor(n_samples / 3);
    end

    random_subject_partition = false;
    if isfield(svm_param.param_search, 'sampling') ...
      && isfield(svm_param.param_search.sampling, 'name') ...
      && strcmp(svm_param.param_search.sampling.name, 'random_subject_partition')
        random_subject_partition = true;
    end
    

    % Split into a training and validation set but use max n_samples
    % samples for the training set to reduce the training time
    split_param = struct();
    %Split into two sets
    split_param.k = 2; 
    % Calculate ratio between training and validation set.
    ratio = max_num_training_samples / n_samples; 
    split_param.ratio = [ratio; 1-ratio];
    if random_subject_partition
        split_param.method = 'random_subjects';
    else
        % If we do stratified or random sampling depends on the data, since
        % stratified is not possible for continuous predictors
        if data.predictor_type(data.predictor_idx) == 0
            split_param.method = 'stratified';
        else
            split_param.method = 'random';
        end
    end

        
    [p,q,r,s,t] = ndgrid(C_range, epsilon_range, gamma_range, degree_range, coef0_range);
    param_matrix = [p(:) q(:) r(:) s(:) t(:)];
    
    perf_matrix = zeros(size(param_matrix,1),trials);
    traintime_matrix = zeros(size(perf_matrix));
    for trial = 1:size(perf_matrix,2)
        % Sample training and validation set
        % Set index for training and validation. New memory is allocated
        % for sample_idx vector only, but not for the features x :)
        data_split = libDataset.split(data, split_param);

        % Try each parameter combination
        for param_idx = 1:size(param_matrix,1)
            % Try param combination
            params = param_matrix(param_idx,:);
            svm_param.C = params(1);
            svm_param.epsilon = params(2);
            svm_param.gamma = params(3);
            svm_param.degree = params(4);
            svm_param.coef0 = params(5);
            
            % Train
            tic;
            model = libSvm.train(data_split{1}, svm_param);
            traintime_matrix(param_idx, trial) = toc;
            
            % Validate
            pred = libSvm.predict(data_split{2}, model, svm_param);
            
            % Calculate performance
            [perf, multiplier] = libPerformance.value(data_split{2}.y(data_split{2}.sample_idx,data_split{2}.predictor_idx), pred, svm_param.param_search.performance_measure);
            perf_matrix(param_idx, trial) = perf * multiplier;
        end
    end
    
    % Select best performance
    mean_perf = mean(perf_matrix,2);
    [~,idx] = sort(mean_perf, 'descend');

    % Set parameters of best performed model
    svm_param.C = param_matrix(idx(1),1);
    svm_param.epsilon = param_matrix(idx(1),2);
    svm_param.gamma = param_matrix(idx(1),3);
    svm_param.degree = param_matrix(idx(1),4);
    svm_param.coef0 = param_matrix(idx(1),5);
   
    
    if any(~isfinite(perf_matrix))
        warning;
    end
    
    % Restore changed model_param values
    if isfield(svm_param.param_search, 'visualize')
        svm_param.visualize = visualize__restore;
    end
    svm_param.param_search.enable = param_search_enable__restore;
    
end

