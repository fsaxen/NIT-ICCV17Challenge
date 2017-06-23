function [ ml_param ] = param_grid_search( data, ml_param )
    % Single predictor only!
    
    ps = ml_param.param_search;

    if ~isfield(ps, 'enable')
        ps.enable = true;
    end
    
    if ps.enable == false
        return;
    end

    if isfield(ps, 'max_num_training_samples')
        max_num_training_samples = ps.max_num_training_samples;
    else
        max_num_training_samples = 1000;
    end
    if isfield(ps, 'max_num_testing_samples')
        max_num_testing_samples = ps.max_num_testing_samples;
    else
        max_num_testing_samples = 10000;
    end
    
    params = ps.params;
    param_matrix = [];
    param_names = cell(0,0);
    for i = 1:2:numel(params)
        name = params{i};
        if ~ischar(name)
            error('ml_param.param_search.params must be a cell array of string and array pairs (ml_param member variable name and test values).');
        end
        val = params{i+1};
        val = val(:);
        n_val = length(val);
        n_pm = max(size(param_matrix,1),1);
        val = reshape(repmat(val',n_pm,1), n_val*n_pm, 1);
        param_matrix = repmat(param_matrix, n_val, 1);
        param_matrix = horzcat(param_matrix, val);
        param_names = horzcat(param_names, name);
    end
    
    

    % avoid recursive call of this function in model_train()
    param_search_enable__restore = ps.enable;
    ml_param.param_search.enable = false;
    
    % do not visualize training during parameter search
    if isfield(ml_param, 'visualize')
        visualize__restore = ml_param.visualize;
        ml_param.visualize = false;
    end
    

    trials = 3;
    n_samples = length(data.sample_idx);
    if n_samples < trials * max_num_training_samples || max_num_training_samples <= 0
        max_num_training_samples = floor(n_samples / trials);
    end
    if max_num_testing_samples + max_num_training_samples > n_samples
        max_num_testing_samples = n_samples - max_num_training_samples;
    end
   

    % Split into a training and validation set but use max n_samples
    % samples for the training set to reduce the training time
    split_param = struct();
    % Split into two sets
    split_param.k = 2; 
    % Calculate ratio between training and validation set.
    ratio = max_num_training_samples / (max_num_training_samples+max_num_testing_samples); 
    split_param.ratio = [ratio; 1-ratio];
    % Select sampling method for split
    if isfield(ps, 'split_sampling')
        split_param.method = ps.split_sampling;
    else
        if isfield(data,'subject')
            % if subject data is avaiable, split randomly without subject
            % overlap
            split_param.method = 'random_subjects';
        elseif data.predictor_type(data.predictor_idx) == 0
            % discrete predictor
            split_param.method = 'stratified';
        else
            % continuous predictor
            split_param.method = 'random';
        end
    end
    
    perf_matrix = zeros(size(param_matrix,1),trials);
    traintime_matrix = zeros(size(perf_matrix));
    for trial = 1:trials
        % Sample training and validation set
        % Set index for training and validation. New memory is allocated
        % for sample_idx vector only, but not for the features x :)
        data_split = libDataset.split(data, split_param);
        
        % If training/validation set is larger than desired, cut it down
        % (may be needed in case of subject split
        cur_num_training_samples = length(data_split{1}.sample_idx);
        if cur_num_training_samples > max_num_training_samples
            data_split{1}.sample_idx = data_split{1}.sample_idx(...
                randperm(cur_num_training_samples, max_num_training_samples));
        end
        cur_num_validation_samples = length(data_split{2}.sample_idx);
        if cur_num_validation_samples > max_num_testing_samples
            data_split{2}.sample_idx = data_split{2}.sample_idx(...
                randperm(cur_num_validation_samples, max_num_testing_samples));
        end

        % Try each parameter combination
        for param_idx = 1:size(param_matrix,1)
            % Set param combination
            ml_param = set_params(ml_param, param_matrix, param_names, param_idx);
            
            % Train
            tic;
            model = libML.train(data_split{1}, ml_param);
            traintime_matrix(param_idx, trial) = toc;
            
            % Validate
            pred = libML.predict(data_split{2}, model, ml_param);
            
            % Calculate performance
            [perf, multiplier] = libPerformance.value(...
                data_split{2}.y(data_split{2}.sample_idx,data_split{2}.predictor_idx),...
                pred, ps.performance_measure);
            perf_matrix(param_idx, trial) = perf * multiplier;
            
            %fprintf('.');
        end
    end
    %fprintf('\n');
    
    % Bug check
    if any(~isfinite(perf_matrix))
        warning('Infinite performance! Bug?');
    end
    
    % Select best performance
    mean_perf = mean(perf_matrix,2);
    [~,idx] = sort(mean_perf);
    
    % To keep color scale distinguishable in interesting range, set worst
    % half of performances to median (cut lower part of scale)
    med_perf = median(mean_perf);
    mean_perf(mean_perf < med_perf) = med_perf;

    % Set parameters of best performed model
    ml_param = set_params(ml_param, param_matrix, param_names, idx(end));
    
    % Plot results
    if isfield(ps, 'filename_prefix') && ~isempty(ps.filename_prefix)
        % Create folder if it does not exist already
        folder = fileparts(ps.filename_prefix);
        if ~isempty(folder) && ~exist(folder, 'dir')
            mkdir(folder);
        end
        % change to log scale if necessary
        for i = 1:size(param_matrix,2)
            param_val = unique(param_matrix(:,i));
            param_val = (param_val - min(param_val)) / range(param_val);
            if abs(median(param_val) - 0.5) > 0.2
                param_matrix(:,i) = log(param_matrix(:,i)) / log(10);
                param_names{i} = [param_names{i}, ' (log10)'];
            end            
        end
        % plot test performance
        fig = figure('Visible','off');
        subplot(2,1,1);
        libUtil.parallelcoords_plus(param_matrix(idx,:), mean_perf(idx), param_names, 1, param_matrix(idx(end),:));
        title_str = 'Test performance (color = ';
        if multiplier ~= 1
            title_str = [title_str, num2str(multiplier), '*'];
        end
        title_str = [title_str, ps.performance_measure, ')'];
        title(title_str, 'Interpreter', 'none');
        % plot training time
        subplot(2,1,2);
        t = mean(traintime_matrix,2);
        [~,t_idx] = sort(t);
        libUtil.parallelcoords_plus(param_matrix(t_idx,:), t(t_idx), param_names, 1, param_matrix(idx(end),:));
        title('Training time (s)');
        drawnow;
        print(fig,'-dpng',sprintf('%s_%d.png', ps.filename_prefix, data.predictor_idx));
        close(fig);        
    end
    
    % Restore changed ml_param values
    if isfield(ml_param, 'visualize')
        ml_param.visualize = visualize__restore;
    end
    ml_param.param_search.enable = param_search_enable__restore;


end

function ml_param = set_params( ml_param, param_matrix, param_names, param_idx )
    param_values = param_matrix(param_idx,:);
    for i = 1:length(param_names)
        assign_cmd = ['ml_param.', param_names{i},'=',num2str(param_values(i)),';'];
        eval(assign_cmd);
    end
end
