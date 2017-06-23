function [ model ] = train( data, ml_param )
%   Train model
%   X: feature matrix (rows are samples)
%   data.y(dataset.sample_idx,predictors_idx): column vector of ground truth response values
%   model_param: type of model, see below for options

    % Use linear model if nothing was set
    if ~isfield(ml_param, 'type')
        ml_param.type = 'lm';
    end
    
    if ~isfield(ml_param, 'num_cpu_cores')
        ml_param.num_cpu_cores = 1;
    end
    
    % Create model structure and set predictor index
    model = struct();
    model.predictor_idx = data.predictor_idx;

    %% Handle multiple predictors
    if length(data.predictor_idx) > 1
        % Multiple predictors are selected. We might need to train a model
        % for each predictor seperately. But this depends on each machine
        % learning method
        switch ml_param.type
            case {'lm','RFc','RFr','SVM','SVR','SVMb','SVMm','EasyEnsemble','Ensemble','ExternalProgram'}
                model.wrap_predictor = true;
            otherwise
                model.wrap_predictor = false;
        end

        if model.wrap_predictor
            
            % Prepare data for training
            n_models = length(data.predictor_idx);
            predictor_model = cell(n_models,1);

            predictor_ml_param = ml_param;
            predictor_ml_param.this_is_a_wraped_predictor = 1; 
                
            
            if ml_param.num_cpu_cores > 1
                % The parallel way
                
                % Save the predictor_idx
                predictor_idx = cell(ml_param.num_cpu_cores, 1);
                predictor_data = cell(ml_param.num_cpu_cores, 1);
                predictor_model_cores = cell(ml_param.num_cpu_cores, 1);
                predictor_ml_param = repmat({predictor_ml_param}, ml_param.num_cpu_cores, 1);

                % If param_search is used, vary filenames to avoid data
                % to be overwritten
                if isfield(ml_param, 'param_search') && isfield(ml_param.param_search, 'filename_prefix')
                    param_search_fn_prefix = ml_param.param_search.filename_prefix;
                else
                    param_search_fn_prefix = '';
                end
                
                % Split predictor_idx
                n_models_per_core = ceil(n_models / ml_param.num_cpu_cores);
                j0 = 1;
                for i = 1 : ml_param.num_cpu_cores
                    j1 = min(j0 + n_models_per_core - 1, n_models);
                    predictor_idx{i} = data.predictor_idx(j0:j1);
                    predictor_model_cores{i} = cell(length(predictor_idx{i}), 1);
                    if ~isempty(param_search_fn_prefix)
                        predictor_ml_param{i}.param_search.filename_prefix = ...
                            sprintf('%s_%d', param_search_fn_prefix, i);
                    end
                    j0 = j1 + 1;
                end
                
                % Reduce dataset to selected samples and predictors (data
                % is copied anyway, but only this way we copy only needed
                % part of it)
                data_sel_samples = libDataset.util_apply_sample_idx(data);
                for i = 1 : ml_param.num_cpu_cores
                    predictor_data{i} = data_sel_samples;
                    predictor_data{i}.predictor_idx = predictor_idx{i};
                    predictor_data{i} = libDataset.util_apply_predictor_idx(predictor_data{i});
                end            
                
                % Start parallelization
                poolobj = gcp('nocreate'); % If no pool, do not create new one.
                if isempty(poolobj)
                    parpool('local',ml_param.num_cpu_cores);
                elseif poolobj.NumWorkers ~= ml_param.num_cpu_cores
                    warning(['Parallel Pool already running with ',num2str(poolobj.NumWorkers),' workers.']);
                end
                
                % Run training in parallel
                t0 = tic;
                parfor i = 1 : ml_param.num_cpu_cores
                %for i = 1 : ml_param.num_cpu_cores
                    pred_idx = predictor_idx{i};
                    pred_data = predictor_data{i};
                    pred_ml_param = predictor_ml_param{i};
                    
                    for j = 1 : length(pred_idx)
                        % tzain
                        pred_data.predictor_idx = j;
                        pred_model = libML.train(pred_data, pred_ml_param);
                        pred_model.predictor_idx = pred_idx(j);
                        predictor_model_cores{i}{j} = pred_model;
                        
                        % report progress
                        if i == 1
                            dt = toc(t0) / j;
                            if dt * length(pred_idx) > 10 && (dt > 5 || mod(j, ceil(5/dt)) == 0)
                                fprintf('libML.train: model %i/%i, %.0f/%.0f seconds\n', ...
                                    j*pred_ml_param.num_cpu_cores, n_models, ...
                                    j*dt, n_models*dt/pred_ml_param.num_cpu_cores);
                            end
                        end
                    end
                end

                % Rearrange trained models
                c = 1;
                for i = 1 : ml_param.num_cpu_cores
                    for j = 1 : length(predictor_model_cores{i})
                        predictor_model{c} = predictor_model_cores{i}{j};
                        c = c + 1;
                    end
                end
                
                fprintf('Total parallel training time: %.1f seconds.\n',toc(t0));
            else
                % The usual way
                t0 = tic;
                
                predictor_data = data;
                
                for i = 1 : n_models
                    
                    predictor_data.predictor_idx = data.predictor_idx(i);
                    
                    % train
                    predictor_model{i} = libML.train(predictor_data, predictor_ml_param);

                    % Output progress
                    dt = toc(t0) / i;
                    if dt * n_models > 10 && (dt > 5 || mod(i, ceil(5/dt)) == 0)
                        fprintf('libML.train: model %i/%i, %.0f/%.0f seconds\n', i, n_models, i*dt, n_models*dt);
                    end
                end
            end
            
            model.predictor_model = predictor_model;
            model.ml_param = ml_param;
            return;
        end
    end
    
    %% -> Only one model (and only one predictor for most learning methods)
        
    % Remove samples with NaN predictor value from training set (NNet can handle NaNs = don't cares)
    if ~strcmp(ml_param.type, 'NNet')
        non_nan_idx = ~isnan(data.y(data.sample_idx, data.predictor_idx));
        data.sample_idx = data.sample_idx(non_nan_idx);
    end
    
    % Run parameter grid search if desired
    if isfield(ml_param, 'param_search')
        ml_param = libML.param_grid_search(data, ml_param);
    end
    
    % Redistribute and subsample data if desired (for Ensemble redistribution is done later, for each ensemble model)
    if ~strcmp(ml_param.type, 'Ensemble')
        data = redistribute_and_subsample(data, ml_param);
    end    
    
    % Train model for one predictor
    switch ml_param.type
        case 'lm' % Linear Regression
            libDataset.util_check_dataset(data, 'y', 'supervised', 'one_predictor_idx', 'normalized');
            % We need to normalize the responces, because fitlm prefers it
            y = data.y(data.sample_idx,data.predictor_idx);
            model.mu_y = mean(y, 1);
            model.std_y = std(y, 1) + eps;
            y = bsxfun(@minus, y, model.mu_y);
            y = bsxfun(@rdivide, y, model.std_y); 
            model.ml = fitlm(data.x(data.sample_idx,:), y);
        case 'RFc' % Random Forest classification
            libDataset.util_check_dataset(data, 'y', 'supervised', 'one_predictor_idx', 'discrete_predictors');
            if ~isfield(ml_param, 'num_trees')
                ml_param.num_trees = 30;
            end
            model.ml = TreeBagger(ml_param.num_trees, ...
                data.x(data.sample_idx,:), ...
                data.y(data.sample_idx,data.predictor_idx));
        case 'RFr' % Random Forest regression
            libDataset.util_check_dataset(data, 'y', 'supervised', 'one_predictor_idx');
            if ~isfield(ml_param, 'num_trees')
                ml_param.num_trees = 30;
            end
            % default: minParent = 10, minLeaf = 1
            model.ml = TreeBagger(ml_param.num_trees, ...
                data.x(data.sample_idx,:), ...
                data.y(data.sample_idx,data.predictor_idx),...
                'method','regression');
        case {'SVM', 'SVR', 'SVMb', 'SVMm'}
            % Dataset will be checked in libSvm.train already
            model.svm = libSvm.train(data, ml_param.svm_param);
        case 'NNet'
            model.nnet = libNNet.train(data, ml_param.nnet_param);
        case 'EasyEnsemble'
            libDataset.util_check_dataset(data, 'y', 'supervised', 'one_predictor_idx', 'discrete_predictors', 'normalized');
            model.ee = libEasyEnsemble.train(data, ml_param.ee_param);
            
        case 'Ensemble'
            % Set standard parameters
            if isfield(ml_param, 'ensemble_num_models')
                n_models = ml_param.ensemble_num_models;
                if n_models < 1 || mod(n_models,1) ~= 0
                    error('ml_param.ensemble_num_models must be a positive integer >= 1.');
                end
            else
                n_models = 4;
            end
            if isfield(ml_param, 'redistribute_param')
                if isfield(ml_param.ensemble_param, 'redistribute_param')
                    warning('Found ml_param.ensemble_param.redistribute_param and ml_param.redistribute_param! Ignoring the latter...');
                else
                    ml_param.ensemble_param.redistribute_param = ml_param.redistribute_param;
                end
            end
            if isfield(ml_param, 'num_samples')
                if isfield(ml_param.ensemble_param, 'num_samples')
                    warning('Found ml_param.ensemble_param.num_samples and ml_param.num_samples! Ignoring the latter...');
                else
                    ml_param.ensemble_param.num_samples = ml_param.num_samples;
                end
            end
            
            % Create ensemble models
            model.ensemble = cell(n_models, 1);
            
            % Train ensemble models
            for i = 1 : n_models
                model.ensemble{i} = libML.train(data, ml_param.ensemble_param);
            end
            
            % Train aggregation model
            data_agg = redistribute_and_subsample(data, ml_param.ensemble_param);
            
            % Predict first ensemble to get the prediction values
            [y1, p1] = libML.predict(data_agg, model.ensemble{1}, ml_param.ensemble_param);
            sy = 1; 
            sp = size(p1, 2);
            if sp == 1 && all(y1 == p1)
                % we do not need p if y == p
                sp = 0;
            end
            syp = sy + sp;
            
            % Allocate memory for aggregation features = ensemble predictions
            x = zeros(length(y1), n_models * syp);
            x(:, 1 : sy) = y1;
            if sp > 0
                x(:, sy + 1 : syp) = p1;
            end
            
            % Create features for aggregation model
            for i = 2 : n_models
                [xi , pi] = libML.predict(data_agg, model.ensemble{i}, ml_param.ensemble_param);
                x(:, (i-1) * syp + 1 : (i-1) * syp + sy) = xi;
                if sp > 0
                    x(:, (i-1) * syp + sy + 1 : i * syp) = pi;
                end
            end
            data_agg = libDataset.create_dataset(x, data_agg.y(data_agg.sample_idx, data_agg.predictor_idx));
            data_agg = libDataset.normalize(data_agg);
            
            % Train aggregation model
            model.aggregation_model = libML.train(data_agg, ml_param.ensemble_param);
            model.aggregation_norm_values = data_agg.norm_values;
            
        case 'ExternalProgram'
            if isfield(ml_param, 'model_fn')
                if size(data.y,2) > 1
                    [dir,name,ext] = fileparts(ml_param.model_fn);
                    model_fn = fullfile(dir, [ name , '_', num2str(data.predictor_idx), ext ]);
                else
                    model_fn = ml_param.model_fn;
                end
            else
                model_fn = [tempname,'.txt'];
            end
            write_not_always = isfield(ml_param, 'only_write_data_on_format_change') && ml_param.only_write_data_on_format_change;
            if isfield(ml_param, 'train_data_fn')
                if write_not_always
                    data_fn = ml_param.train_data_fn;
                else
                    data_fn = [ml_param.train_data_fn '_' num2str(length(data.sample_idx)) 'x' num2str(size(data.x,2) + length(data.predictor_idx)) '.txt'];
                end
            else
                data_fn = [tempname,'.txt'];
            end
            % write data to file
            if ~write_not_always || ~exist(data_fn, 'file')
                train_data = horzcat(data.x(data.sample_idx,:), data.y(data.sample_idx,data.predictor_idx));
                dlmwrite(data_fn, train_data, '\t');
            end
            % run training program
            cmd = strrep(ml_param.train_command_line, '$model_fn', model_fn);
            
            cmd = strrep(cmd, '$data_fn', data_fn);
            system(cmd);
            model.model_fn = model_fn;
        otherwise
            error(strcat('Training type: ', ml_param.type, ' not supported yet.'));
    end
    
    % Safe ml_param on top level (not wraped predictors)
    if ~isfield(ml_param, 'this_is_a_wraped_predictor')
        model.ml_param = ml_param;
    end
    
    
    %% private functions
    
    function data = redistribute_and_subsample(data, ml_param)
        
        % Redistribute dataset if desired 
        if isfield(ml_param, 'redistribute_param')
            data = libDataset.redistribute(data, ml_param.redistribute_param);
        end

        % Subsample data if desired
        if isfield(ml_param, 'num_samples')
            n_curr = length(data.sample_idx);
            n_wanted = ml_param.num_samples;
            if n_wanted <= 1
                % <= 1 means relative sample count
                n_wanted = round(n_wanted * n_curr);
            end
            split_param = struct('k', 1, 'ratio', min(1, n_wanted/n_curr));
            if data.predictor_type(data.predictor_idx) == 0 % 'discrete' predictor
                split_param.method = 'stratified';
            else
                split_param.method = 'random';
            end
            data_subsample = libDataset.split(data, split_param);
            data = data_subsample{1};
        end
        
    end

    
end
