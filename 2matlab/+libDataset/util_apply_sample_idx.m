function data = util_apply_sample_idx(data)

% Apply Shrink dataset features and labels to length(dataset.sample_idx).
% Be aware that new memory will be allocated.

    % features
    data.x = data.x(data.sample_idx, :);
    
    % labels / predictors
    if isfield(data, 'y')
        data.y = data.y(data.sample_idx, :);
    end
    
    % subject
    if isfield(data, 'subject')
        data.subject = data.subject(data.sample_idx);
    end

    % filename
    if isfield(data, 'filename')
        data.filename = data.filename(data.sample_idx);
    end
    
    % Set new sample_idx
    data.sample_idx = (1 : length(data.sample_idx))';
end