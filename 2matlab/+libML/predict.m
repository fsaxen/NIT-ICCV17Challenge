function [ y, p ] = predict( data, model, ml_param )
% Predict response from features
%
% [ y, p ] = predict( data, model )
%   model: model returned by ml.train
%   dataset: Struct containing 
%       .x = Entire feature dataset (one row per sample)
%       .sample_idx = Index of samples to predict.
%
%   y: Output responses (binary or regression)
%
%   p: Ensemble scores or SVM prediction matrix (or empty)
%
% [ y, p ] = predict( data, model, ml_param )
%
%   ml_param: passing parameters for wraped models.
%

    % Check input
    if ~isstruct(model)
        error('Input model must be created by +libML.train.');
    end

    % If ml_param is not given, use from model
    if nargin < 3
        ml_param = model.ml_param;
    end

    % Check if dataset is legit
    if ~isfield(ml_param, 'skip_data_check') || ~ml_param.skip_data_check
        libDataset.util_check_dataset(data);
        % we can safely skip check in child calls
        ml_param.skip_data_check = 1;
    end
    
    % Most classifiers don't provide scores :(
    p = [];

    % Are there several predictors and we got to wrap them up?
%     % Unfortunately, we can not provide the scores because they are
%     % inconsistent from one classifier to another.
    num_predictors = length(model.predictor_idx);
    if num_predictors > 1 && model.wrap_predictor
        y = zeros(size(data.sample_idx,1), num_predictors);
        p = y;
        ps = 1;
        for i = 1 : num_predictors
            data.predictor_idx = model.predictor_idx(i);
            [y(:,i), pi] = libML.predict(data, model.predictor_model{i}, ml_param);
            if nargout > 1
                if i == 1 && size(pi, 2) ~= 1 % Adjust p if the probs don't exist or have more dimensions than just one.
                    ps = size(pi, 2);
                    p = zeros(size(data.sample_idx,1), num_predictors * ps);
                end
                p(:, (i-1)*ps+1 : i*ps) = pi;
            end
        end
        return;
    end



    % Apply primary model
    switch ml_param.type
        case {'SVM', 'SVMm', 'SVR', 'SVMb'}
            [y, p] = libSvm.predict(data, model.svm, ml_param.svm_param);
        case 'NNet'
            [y, p] = libNNet.predict(data, model.nnet, ml_param.nnet_param);
        case 'EasyEnsemble'
            [y, ~, p] = libEasyEnsemble.predict(data, model.ee);
        case {'Ensemble'}
            
            n_samples = size(data.sample_idx, 1);
            n_models = length(model.ensemble);
            
            % Predict first ensemble to get the prediction values
            [y1, p1] = libML.predict(data, model.ensemble{1}, ml_param.ensemble_param);
            sy = size(y1, 2); 
            sp = size(p1, 2);
            if sp == 1 && all(y1 == p1)
                % we do not need p if y == p
                sp = 0;
            end
            syp = sy + sp;
            
            % Allocate memory for ensemble scores
%             p = zeros(n_samples, n_models * sy);
%             p(:, 1:sy) = y1;
            
            % Allocate memory for ensemble predictions
            yy = zeros(n_samples, n_models);
            
            for i = 1 : n_models
                yy(:,i) = libML.predict(data, model.ensemble{i}, ml_param.ensemble_param);
            end
            
            % majority voting:
            p = mean(yy,2) - 0.5;
            y = p > 0;
        case 'lm'
            y = predict(model.ml, data.x(data.sample_idx, :));
            % We need to denormalize y because the linear model was fed
            % with normalized responces
            y = bsxfun(@times, y, model.std_y); 
            y = bsxfun(@plus, y, model.mu_y);
            
        case 'matlab_nn'
            [y] = model.net(data.x(data.sample_idx, :)')';
            
        case 'ExternalProgram'
            data_fn = [tempname,'.txt'];
            pred_fn = [tempname,'.txt'];
            % write data to file
            test_data = horzcat(data.x(data.sample_idx,:), data.y(data.sample_idx,data.predictor_idx));
            dlmwrite(data_fn, test_data, '\t');
            % run training program
            cmd = strrep(ml_param.predict_command_line, '$model_fn', model.model_fn);
            cmd = strrep(cmd, '$data_fn', data_fn);
            cmd = strrep(cmd, '$pred_fn', pred_fn);
            system(cmd);
            % read predictions
            pred_data = dlmread(pred_fn, '\t');
            y = pred_data(:,2);
            
        otherwise
            % Use matlab buildin prediction
            [y, p] = predict(model.ml, data.x(data.sample_idx, :));
            if iscell(y)
                y = str2double(y);
            end
    end
    
end
