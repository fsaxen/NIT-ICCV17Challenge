function data = normalize(data, norm_values)
% Normalize entire dataset
%
%   data = normalize(data) calculates the norm_values and normalizes data
%   according to it.
%
%   data = normalize(data, norm_values) uses the given norm values to
%   normalize data.
%
%   data =
%       Dataset structure. See libDataset.create_dataset() for details.
%
%   norm_values =
%       Structure containing mu_x, std_x (and mu_y, and std_y if data.y
%       exist). Each vector must have the same number of colums as data.x
%       (data.y respectively for mu_y and std_y) and one row.
%

    % Check dataset consistency first
    libDataset.util_check_dataset(data);
    
    % Check if dataset has already been normalized
    if isfield(data, 'norm_values')
        warning('Dataset has already been normalized.');
        return;
    end
    
    % Check input parameters
    num_features = size(data.x,2);

    % Calculate norm_values
    if nargin == 1 || ~isstruct(norm_values)
        % Calculate norm values
        norm_values = struct();
        
        % Features
        % Initialize mean and std values
        norm_values.mu_x = nanmean(data.x,1); % Mean value of each feature
        norm_values.std_x = zeros(1, num_features);
        % Calculate std value for each feature independently if the amount
        % of data is huge.
        if numel(data.x) > 1e7
            for f = 1 : num_features
                norm_values.std_x(f) = nanstd(data.x(:,f));
            end
        else
            norm_values.std_x = nanstd(data.x,0,1); % Std deviation of each feature
        end
        
        % Prevent from dividing by 0
        norm_values.std_x = norm_values.std_x + eps;
    end
    
    % Normalize dataset
    % Features
    data.x = bsxfun(@minus, data.x, norm_values.mu_x);
    data.x = bsxfun(@rdivide, data.x, norm_values.std_x);
    
    % Add norm_values to dataset
    data.norm_values = norm_values;
end