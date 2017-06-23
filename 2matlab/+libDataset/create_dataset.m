function dataset = create_dataset(x, varargin)
% dataset = create_dataset(x)
% dataset = create_dataset(x, y)
% dataset = create_dataset(x, y, subject)
% dataset = create_dataset(x, y, subject, filename)
% dataset = create_dataset(..., name, value)
%
%   Create a dataset.
%   name                value
%   'predictor_type'    array of length size(y,2)
%                       'discrete'
%                       'continuous'
%                       'unsupervised'
%
% dataset
%   .x =
%       Features, where each sample fills one row of x
%
%   .sample_idx =
%       Vector of indices corresponding to samples in x, y, etc. In this
%       dataset implementation, you usually don't copy any features or
%       labels when you split the dataset e.g. into a training and
%       validation set. Instead, you just change the sample_idx vector to
%       those samples that shall be trained or validated respectively. This
%       is especially powerful for very large dataset, because matlab does
%       only copy matrices if they differ and as long as sample_idx
%       differs, the features remain untouched.
%       NO LOGICAL INDEX SUPPORTED!
%
%   .feature_idx =
%       TODO -> Not yet implemented but necessary for feature selection
%
%   .y =
%       Labels / predictors of x. Must have the same number of rows like x
%       and one predictor for each column. Please specify your predictor in
%       predictor_type. y does not need to exist if you have an
%       unsupervised learning problem. Just set predictor_type to
%       'unsupervised'.
%   
%   .predictor_type = 
%       array of length size(y,2). Corresponding to each predictor, set
%       value to 0 for 'discrete', 1 for 'continuous'. If you have an
%       unsupervised learning problem, you do not have to provide y, but
%       must set predictor_type to the value (not array) 2.
%
%   .predictor_idx
%       Vector of indices pointing to active predictor(s) in y. Each
%       predictor in y. Also not necessary for unsupervised learning
%       problems.
%
%   .subject
%       Vector of integers with the same number of rows like x. Each row in
%       subject provides the subject number (integer) for that particular
%       sample.
%
%   .filename
%       Cell Vector of strings providing the filename for each sample. Each
%       row in filename provides the filename (string) for that particular
%       sample.
%
%   .norm_values
%       Structure containing the normalization values for the features and 
%       labels: mu_x, mu_y, std_x, std_y. Calculated and set by
%       libDataset.normalize().
%
    p = inputParser;
    expectedPredTypes = {'discrete','continuous','unsupervised'};

    addRequired(p,'x',@isnumeric);
    addOptional(p,'y',[],@isnumeric);
    addOptional(p,'subject',[],@isnumeric);
    addOptional(p,'filename',[],@iscell);
    addParameter(p,'predictor_type',[],...
                 @(x) isnumeric(x) || any(validatestring(x,expectedPredTypes)));

    parse(p,x,varargin{:});
    p = p.Results;

    dataset = struct();
    if ~isempty(p.x)
        % Set features
        dataset.x = p.x;
        num_samples = size(p.x, 1);
        dataset.sample_idx = (1 : num_samples)';
        dataset.predictor_type = 2;
    end
    
    if ~isempty(p.y)
        if size(p.y,1) ~= size(p.x, 1)
            error('Number of samples in x and y must match.');
        end
        % Set labels
        dataset.y = p.y;
        num_predictors = size(p.y, 2);
        dataset.predictor_idx = (1 : num_predictors)';

        % Calculate if the predictors are continuous or discrete
        if isempty(p.predictor_type)
            y_no_nan = p.y; y_no_nan(isnan(y_no_nan(:))) = 0;
            is_integer = all((y_no_nan - floor(y_no_nan)) == 0, 1);
            dataset.predictor_type = ~is_integer(:); % 1='continuous', 0='discrete'
            if all(dataset.predictor_type == 0)
                % All predictors are discrete
                %disp('create_dataset: discrete predictors.');
            elseif all(dataset.predictor_type == 1)
                % All predictors are continuous
                %disp('create_dataset: continuous predictors.');
            else
                % Mixed
                warning('Mixed predictors in dataset.');
            end
        end
    end
    
    if ~isempty(p.predictor_type)
        % Set predictor type
        if isnumeric(p.predictor_type)
            if any(p.predictor_type(:) == 2)
                % unsupervised
                if ~all(p.predictor_type(:) == 2)
                    error('You cannot mix supervised and unsupervised predictor_type.');
                end
                dataset.predictor_type = 2;
            elseif size(p.y,2) == numel(p.predictor_type)
                % supervised
                if any(p.predictor_type(:) < 0 | p.predictor_type(:) > 2)
                    error('Value of predictor_type must be 0 (discrete), 1 (continuous), or 2 (unsupervised).');
                end
                dataset.predictor_type = p.predictor_type(:);
            else
                error('Number of predictors in y and predictor_type must match.');
            end
        else
            switch lower(p.predictor_type)
                case 'discrete'
                    dataset.predictor_type = zeros(num_predictors, 1);
                case 'continuous'
                    dataset.predictor_type = ones(num_predictors, 1);
                case 'unsupervised'
                    dataset.predictor_type = 2;
                otherwise
                    error('Unknown predictor_type.');
            end
        end
    end
    
    % Subject and/or filename might not exist
    if ~isempty(p.subject)
        % The number of rows in subject must match the number of samples
        if size(p.subject, 1) ~= size(p.x, 1)
            error('The number of rows in subject must match the number of samples.');
        end
        if size(p.subject, 2) ~= 1
            error('The number of columns in subject must be 1.');
        end
        
        dataset.subject = p.subject;
    end
    if ~isempty(p.filename)
        % Filename must be a cell array
        if ~iscell(p.filename)
            error('Filename must be a cell array of size numSamples x 1. It contains one filename for each row which corresponds to the appropriate sample.');
        end
        % The number of rows in filename must match the number of samples
        if size(p.filename, 1) ~= size(p.x, 1)
            error('The number of rows in filename must match the number of samples.');
        end
        if size(p.filename, 2) ~= 1
            error('The number of columns in filename must be 1.');
        end
        
        dataset.filename = p.filename;
    end

end