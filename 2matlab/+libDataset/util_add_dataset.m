function data = util_add_dataset(varargin)
% data = util_add_dataset(set1, set2, set3, ...)
% data = util_add_dataset(set_cell_array)
% data = util_add_dataset('NoCopyCheck', ...)
%
% Combine an arbitrary number of datasets.
% If the datasets do only differ in sample_idx, only sample_idx will be
% combined withtout copying the features etc.
% If the datasets features differ significantly, the data and all other
% attributes will be copied. But there are several limitations:
% 1. The number of features for each dataset must match
% 2. The number of predictors must match
% 3. The predictor types must match
% 4. If the data is normalized, the normalization parameters must match
% 5. The predictor_idx must match
% 6. The feature_idx must match

    % If second dataset is empty, just return data_a
    if nargin == 0
        error('No dataset to combine.');
    end

    % Handle if first argument is string
    no_copy_check = false;
    if ischar(varargin{1})
        if strcmpi(varargin{1}, 'NoCopyCheck')
            % check if we need to copy data may be very expensive, so offer option to skip this
            no_copy_check = true;
        end
        varargin = varargin(2:end);
        %nargin = nargin - 1;
    end
    
    % You can provide each set independently in the argument list or you
    % can provide a cell of sets.
    if length(varargin) == 1 && iscell(varargin{1})
        % 1 cell of datasets
        set = varargin{1};
        num_sets = length(set);
    else
        % Get sets from argument list
        set = varargin;
        num_sets = length(varargin);
    end
    
    % Get access to the first dataset
    data = set{1};
    
    % Check dataset consistency first
    libDataset.util_check_dataset(data);
    data_sampled = false;
    
    for i = 2 : num_sets
        % Take next dataset
        data_i = set{i};
        
        % Check dataset consistency
        libDataset.util_check_dataset(data_i);
        
        % 1. Check if the features match
        if size(data_i.x,2) ~= size(data_i.x,2)
            error(strcat('Number of features must match. Dataset ', num2str(i), ' failed.'));
        end
        
        % 2. Check if the number of predictors match
        if isfield(data, 'y') && size(data.y, 2) ~= size(data_i.y,2)
            error(strcat('Number of predictors must match. Dataset ', num2str(i), ' failed.'));
        end
        
        % 3. Check if the predictor types match
        if any(data.predictor_type ~= data_i.predictor_type)
            error(strcat('Predictor types must match. Dataset ', num2str(i), ' failed.'));
        end
        
        % 4. Check normalization parameters
        if xor(isfield(data, 'norm_values'), isfield(data_i, 'norm_values'))
            error('One dataset is normalized but not the other.');
        elseif isfield(data, 'norm_values') && isfield(data_i, 'norm_values')
            % Both datasets provide normalization paramters -> Check
            % consistency
            % Check mu_x and std_x
            norm_match = max(abs(data.norm_values.mu_x - data_i.norm_values.mu_x)) < 1e-6;
            norm_match = norm_match && max(abs(data.norm_values.std_x - data_i.norm_values.std_x)) < 1e-6;
            % Check mu_y and std_y if they exist
            if isfield(data.norm_values, 'mu_y')
                norm_match = norm_match && max(abs(data.norm_values.mu_y - data_i.norm_values.mu_y)) < 1e-6;
                norm_match = norm_match && max(abs(data.norm_values.std_y - data_i.norm_values.std_y)) < 1e-6;
            end
            % Did anything not match?
            if ~norm_match
                error(strcat('Normalization parameters of datasets do not match. Dataset ', num2str(i), ' failed.'));
            end
        end
        
        % 5. Also the predictor_idx must match
        if isfield(data, 'y') && sum(abs(sort(data.predictor_idx) - sort(data_i.predictor_idx))) ~= 0
            error(strcat('Predictor_idx does not match. Dataset ', num2str(i), ' failed.'));
        end
        
        % 6. Check if feature_idx match
        if xor(isfield(data, 'feature_idx'), isfield(data_i, 'feature_idx'))
            error(strcat('feature_idx is present for one dataset but not for the other. Dataset ', num2str(i), ' failed.'));
        elseif isfield(data, 'feature_idx') && sum(abs(data.feature_idx - data_i.feature_idx)) > numel(data.feature_idx) / 1e-7
            error(strcat('feature_idx differs between datasets. Dataset ', num2str(i), ' failed.'));
        end            
        

        % Do we need to copy data?
        copy = false;
        if size(data.x,1) ~= size(data_i.x,1)
            copy = true;
        elseif ~no_copy_check && any(abs(data.x(:) - data_i.x(:)) > 1e-7)   % may be expensive!!
            copy = true;
        end
        
        if copy
            % We need to allocate new memory -> Apply sample_idx first
            % For data, we only have to do that once!
            if ~data_sampled
                data = libDataset.util_apply_sample_idx(data);
                data_sampled = true;
            end
            % Apply sample_idx for data_i
            data_i = libDataset.util_apply_sample_idx(data_i);
            
            % Append each dataset field
            data.x = [data.x; data_i.x];
            
            % Because we applied sample_idx, we simply can generate a new
            % one.
            data.sample_idx = (1 : length(data.sample_idx) + length(data_i.sample_idx))';
            
            % Add labels if they exist
            if isfield(data, 'y')
                data.y = [data.y; data_i.y];
            end
            
            % Add subjects if they exist
            if isfield(data, 'subject')
                data.subject = [data.subject; data_i.subject];
            end
            
            % Add filenames if they exist
            if isfield(data, 'filename')
                data.filename = [data.filename; data_i.filename];
            end
            
        else
            % Just append sample_idx
            data.sample_idx = [data.sample_idx; data_i.sample_idx];
        end
        
    end    
end