function [ performance, runtime_info, predictions, scores, predictions_idx] = cross_validate( data, cross_validate_param)
% 

    % Check dataset
    libDataset.util_check_dataset(data, 'supervised');
    % Normalize dataset
    data = libDataset.normalize(data);

    % Check cross_validate_param
    if ~isfield(cross_validate_param, 'ml_param')
        error('Please specify a machine learning method in cross_validate_param.ml_param.');
    end
    
    % Set standard parameters
    if ~isfield(cross_validate_param, 'k')
        cross_validate_param.k = 10;
    end
    if ~isfield(cross_validate_param, 'split_method')
        cross_validate_param.split_method = 'random';
    end
    if ~isfield(cross_validate_param, 'num_folds')
        cross_validate_param.num_folds = cross_validate_param.k;
    end
    if ~isfield(cross_validate_param, 'performance_measure')
        cross_validate_param.performance_measure = 'mse';
    end
    
    % Check parameters
    if cross_validate_param.num_folds > cross_validate_param.k
        error('The variable cross_validate_param.num_folds must be smaller than (or equal to) cross_validate_param.k.');
    end
    
    % Split dataset
    split_param = struct();
    split_param.k = cross_validate_param.k;
    split_param.method = cross_validate_param.split_method;
    sets = libDataset.split(data, split_param);

    
    % Set standard parameters
    num_folds = cross_validate_param.num_folds;
    perf = cross_validate_param.performance_measure;
    
    % How many performance measures do we have?
    if iscell(perf)
        num_perf = length(perf);
    else
        % Convert performance measure into cell
        num_perf = 1;
        perf_n = cell(1,1);
        perf_n{1} = perf;
        perf = perf_n;
    end
    
    % Allocate memory for output
    num_valid_predictors = length(data.predictor_idx);
    performance_int = zeros(num_folds, num_perf, num_valid_predictors);
    runtime_info = struct();
    runtime_info.training_runtime = zeros(num_folds, 1);
    runtime_info.testing_runtime = zeros(num_folds, 1);
    num_predictors = length(data.predictor_idx);
    % Calculate number of output samples
    num_samples = 0;
    for k = 1 : num_folds
        num_samples = num_samples + size(sets{k}.sample_idx, 1);
    end
    predictions = zeros(num_samples, num_predictors);
    scores = zeros(num_samples, num_predictors);
    predictions_idx = zeros(num_samples, 1);
    pred_counter = 1;

    % Train split_param.k models and evaluate them
    t0 = tic;
    for k = 1 : num_folds
        % Use kth set to validate
        testing_set = sets{k};
        % Use all other sets to train
        training_set = libDataset.util_add_dataset(sets{[1:k-1, k+1:split_param.k]});
        
        % Train model on training set
        t1 = tic;
        model = libML.train(training_set, cross_validate_param.ml_param);
        runtime_info.training_runtime(k) = toc(t1);
        
        % Test model on testing set
        t1 = tic;
        [pred, prob] = libML.predict(testing_set, model, cross_validate_param.ml_param);
        runtime_info.testing_runtime(k) = toc(t1);
        
        % Measure performance
        performance_int(k,:,:) = libPerformance.value(testing_set.y(testing_set.sample_idx, testing_set.predictor_idx), pred, perf);
        
        % Set output predictions
        predictions(pred_counter : pred_counter + size(pred,1) - 1, :) = pred;
        predictions_idx(pred_counter : pred_counter + size(pred,1) - 1, :) = testing_set.sample_idx;
        
        % Set output probabilities
        if k == 1 && size(prob, 2) ~= num_predictors
            scores = zeros(num_samples, size(prob, 2));
        end
        scores(pred_counter : pred_counter + size(pred,1) - 1, :) = prob;
        
        % Prepare for next iteration
        pred_counter = pred_counter + size(pred,1);
    end
    
    runtime_info.total_runtime = toc(t0);
    
    % Average performance measure across the num_folds folds and convert it
    % into cell type
    performance = cell(num_perf + 1, num_valid_predictors + 1);
    performance{1,1} = 'Performance measure in each row, predictor idx in the columns.';
    % Write performance measure into cell
    for i = 1 : num_perf
        performance{i + 1, 1} = perf{i};
    end
    % Write predictor idx into first row
    for j = 1 : num_valid_predictors
        performance{1, j + 1} = data.predictor_idx(j);
    end
    % Average performance
    performance_int = mean(performance_int, 1);
    
    % Write performance into cell matrix
    for i = 1 : num_perf
        for j = 1 : num_valid_predictors
            performance{i + 1, j + 1} = performance_int(1, i, j);
        end
    end
    
end
