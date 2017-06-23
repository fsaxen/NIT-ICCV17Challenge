function [performance, runtime_info, predictions, predictions_idx] = cross_db_validate( varargin )
%cross_db_validate - Performs a cross database evaluation
%
%
    % Check input
    if nargin < 2
        error('Invalid input data. Did you forgot to provide cross_db_validate_param?');
    end

    % The last parameter must be the cross_db_validate_param
    cross_db_validate_param = varargin{nargin};
    param = cross_db_validate_param;
    
    % Test input
    if ~isstruct(param)
        error('The last parameter must be the cross_db_validate_param.');
    end
    if ~isfield(param, 'ml_param')
        error('Please specify a machine learning method in cross_db_validate_param.ml_param.');
    end
    if ~isfield(param, 'performance_measure')
        error('Please specify a performance measure in cross_db_validate_param.performance_measure.');
    end
    
    % Set standard parameter
    if isfield(param, 'load_from_file') && param.load_from_file
        param.load_from_file = true;
    else
        param.load_from_file = false;
    end
    
    % Check input consistency
    if param.load_from_file && nargin ~= 2
        error('You must provide 2 input arguments if you want to load the datasets from file.');
    end
    
    % Calculate number of input datasets
    if param.load_from_file
        db_cell = varargin{1};
        num_dbs = size(db_cell, 1);
    else
        num_dbs = nargin - 1;
    end
    
    % 1 dataset is not enough
    if num_dbs < 2
        error('You must provide at least 2 datasets to do cross database validation');
    end
    
    % Allocate output variables
    performance = cell(num_dbs);
    runtime_info = cell(num_dbs);
    predictions = cell(num_dbs);
    predictions_idx = cell(num_dbs);
        
    % Train each ith dataset with each jth dataset
    for i = 1 : num_dbs
        
        % Get dataset i and j
        if param.load_from_file
            % Load dataset i from file
            % dataset i
            db_filename = db_cell{i,1};
            db_varname = db_cell{i,2};
            load(db_filename, db_varname);
            eval(strcat('dbi = ',db_varname,';'));
            eval(['clear ', db_varname, ';']);
        else
            % The standard: All datasets are in the argument list
            dbi = varargin{i};
        end
           
        % Check datasets
        libDataset.util_check_dataset(dbi, 'supervised');

        % Normalize datasets if necessary
        dbi = libDataset.normalize(dbi);

        % Train on dataset i
        t0 = tic;
        model = libML.train(dbi, param.ml_param);
        training_runtime = toc(t0);
       
        % Free memory of training set
        clear dbi;
        
        for j = 1 : num_dbs
%             % We do not train and test the same db (i == j)
%             if i == j
%                 continue;
%             end
            
            % Get dataset j
            if param.load_from_file
                % dataset j
                db_filename = db_cell{j,1};
                db_varname = db_cell{j,2};
                load(db_filename, db_varname);
                eval(strcat('dbj = ',db_varname,';'));
                eval(['clear ', db_varname, ';']);
            else
                % The standard: All datasets are in the argument list
                dbj = varargin{j};
            end
            
            % Prepare time value
            time_info = struct();
            time_info.training_time = training_runtime;
            
            % Check dataset
            libDataset.util_check_dataset(dbj, 'supervised');
            
            % Normalize datasets if necessary
            dbj = libDataset.normalize(dbj);
            
            % Validate on db j
            t0 = tic;
            y = libML.predict(dbj, model, param.ml_param);
            time_info.testing_runtime = toc(t0);
            
            % Calculate performance
            perf = libPerformance.value(dbj.y(dbj.sample_idx, dbj.predictor_idx), y, param.performance_measure);
            
            % Set output values
            performance{i,j} = perf;
            runtime_info{i,j} = time_info;
            predictions{i,j} = y;
            predictions_idx{i,j} = dbj.sample_idx;
            
            % Free momeory of datasets
            clear dbj;
        end
    end

end
