function sets = split(data, split_param)
% split dataset into split_param.k sets
%   data =
%       dataset structure. Use libDataset.create_dataset() to create a
%       valid dataset structure.
%
%   split_param =
%       Splitting parameters. Please see libDataset.spit_param.m for
%       details.
%
    
    import libDataset.*

    % Check input
    if nargin < 2
        warning('No split parameters provided. Using standard parameters.');
        split_param = struct();
    elseif nargin < 1
        error('Providing the dataset is mandatory.');
    end
    
   
    % Set standard parameters.
    if ~isfield(split_param, 'method')
        split_param.method = 'random';
    end
    
    if ~isfield(split_param, 'k')
        split_param.k = 2;
    elseif split_param.k <= 0 || mod(split_param.k,1) ~= 0
        error('You must provide a positive integer for split_param.k');
    end
    
    if ~isfield(split_param, 'ratio')
        split_param.ratio = 1 / split_param.k;
    end
    
    if ~isfield(split_param, 'with_repetition')
        split_param.with_repetition = false;
    end
    
    
    % Check split_param input
    if any(split_param.ratio < 0) || (~split_param.with_repetition && sum(split_param.ratio) > 1)
        error('Each element of split_param.ratio must be > 0 and the sum of ratios must not exceed 1 if we sample without repitition.');
    end
    
    if length(split_param.ratio) == 1
        split_param.ratio = ones(split_param.k, 1) * split_param.ratio;
    end
    
    if length(split_param.ratio) ~= split_param.k
        error('split_param.ratio must be a scalar or a vector with split_param.k elements');
    end
    
    % Perform dataset splitting (checking the dataset consistency is done
    % in each splitting method separately)
    switch split_param.method
        case 'random'
            sets = split_random(data, split_param);
        case 'stratified'
            sets = split_stratified(data, split_param);
        case 'random_subjects'
            sets = split_random_subjects(data, split_param);
        otherwise
            error('Unkown splitting method.');
    end
end