function util_check_dataset(varargin)

    if nargin < 1
        error('To check the dataset, you must at least provide the dataset.');
    end
    
    data = varargin{1};
    
    % Check mandatory elements of dataset
    if ~isstruct(data)
        error('Dataset must be a struct.');
    end
    
    % predictor_type must exist and provide 'discrete', 'continuous'
    % elements only. Or it contains one element with the value 'unsupervised'.
    if ~isfield(data, 'predictor_type')
        error('You must provide dataset.predictor_type, containing values for discrete (0) or continuous (1) predictors, depending on the predictor in each dataset.y column. If you have an unsupervised learning problem, set predictor_type to unsupervised.');
    else % Check each element. Only 'discrete' and 'continous' is allowed
        if isscalar(data.predictor_type) && data.predictor_type == 2
            supervised = false;
        else
            supervised = true;
            num_predictors = length(data.predictor_type);
            s_discrete = sum(data.predictor_type == 0);
            s_continuous = sum(data.predictor_type == 1);
            discrete = s_discrete == num_predictors;
            continuous = s_continuous == num_predictors;
            mixed = (s_discrete + s_continuous) == num_predictors;
            if ~mixed && ~continuous && ~discrete
                error('Invalid predictor type found. dataset.predictor_type must be continuous (1) or discrete (0) for each predictor.');
            end
        end
    end
    
    % sample index exists
    if ~isfield(data, 'sample_idx') || isempty(data.sample_idx) || size(data.sample_idx, 2) ~= 1
        error('data.sample_idx is not present or empty or contains more than 1 column.');
    end
    
    % features must exists
    if ~isfield(data, 'x') || isempty(data.x)
        error('No features found. data.x not existing or empty.');
    end
    
    % predictor_idx exists if supervised learning
    if supervised && (~isfield(data, 'predictor_idx') || isempty(data.predictor_idx))
        error('No predictor index found. dataset.predictor_idx not existing or empty.');
    end
    
    % predictor_idx points to a valid predictor
    if supervised
        predictor_inval = data.predictor_idx < 1 ...
            | data.predictor_idx > size(data.y, 2) ...
            | floor(data.predictor_idx) ~= data.predictor_idx;
        if any(predictor_inval)
            error('dataset.predictor_idx points to a not existing predictor in dataset.y.');
        end
    end
   
    % Additional checks by vargin
    for v = 2 : nargin
        check = varargin{v};
        
        switch check
            
            case 'supervised' % Learning problem must be supervised
                if ~supervised
                    error('Dataset learning problem is unsupervised.');
                end
                
            case 'y' % Check if the predictor is present
                % predictors, labels must exists if supervised learning problem
                if supervised && (~isfield(data, 'y') || isempty(data.y))
                    error('No predictors found. data.y not existing or empty.');
                end

                % Number of predictors must match
                if supervised && size(data.y,2) ~= length(data.predictor_type)
                    error('For supervised learning problems you must provide for each predictor in y (each column in y) a predictor_type element.');
                end

                % features and predictors must match in size if supervised learning
                if supervised && (size(data.x, 1) ~= size(data.y, 1))
                    error('Number of rows in dataset.x must match number of rows in dataset.y.');
                end
            
            case 'continuous_predictors' % Check if the predictors are continuous (regression) values, not discrete (classification) labels.
                if ~supervised || sum(data.predictor_type(data.predictor_idx) == 1) ~= length(data.predictor_idx)
                    error('The called function can only handle continuous (regression) values, not discrete (classification) labels or unsupervised data.');
                end
                
            case 'discrete_predictors' % Check if the predictors are discrete (classification) values, not continous (regression) values.
                if ~supervised || sum(data.predictor_type(data.predictor_idx) == 0) ~= length(data.predictor_idx)
                    error('The called function can only handle discrete labels, not continuous regression values or unsupervised data.');
                end
                
            case 'filename' % Check if the filename field is present and does provide the same number of samples
                if ~isfield(data, 'filename') || size(data.filename, 1) ~= size(data.x, 1) || size(data.filename, 2) ~= 1
                    error('The filename field is not present or doesnt provide the same number of samples like x or has more than 1 column.');
                end
                
            case 'one_predictor_idx' % Check if there is only one chosen predictor, because the function can not deal with multiple predictors.
                if ~isscalar(data.predictor_idx) || data.predictor_idx <= 0
                    error('data.predictor_idx is not a scalar but the called function can only deal with a single predictor.');
                end
                
            case 'subject' % Check if the subject field is present and does provide the same number of samples
                if ~isfield(data, 'subject') || size(data.subject, 1) ~= size(data.x, 1) || size(data.subject, 2) ~= 1
                    error('The subject field is not present or doesnt provide the same number of samples like x or has more than 1 column.');
                end
                
            case 'unsupervised' % Learning problem must be unsupervised
                if supervised
                    error('Dataset learning problem is supervised.');
                end
                
            case 'valid_sample_idx' % Check if all sample_idx point to existing samples (expensive to check)
                for sample = data.sample_idx
                    if rem(sample, 1) ~= 0 || sample < 0 || sample > size(data.x, 1)
                        error('data.sample_idx points to an not existing sample in data.x.');
                    end
                end
                
            case 'normalized' % Check if dataset is normalized
                if ~isfield(data, 'norm_values')
                    error('Dataset is not normalized.');
                elseif length(data.norm_values.mu_x) ~= size(data.x, 2) || length(data.norm_values.std_x) ~= size(data.x, 2)
                    error('Number of element for mu_x and/or std_x do not match with number of features in x.');
                end
                
            otherwise
                error(strcat('Unkown check parameter: ', check));
        end
    end
end