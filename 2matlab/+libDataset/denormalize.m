function x = denormalize(x, mu_x, std_x)
%denormalize - Denormalize a normalized dataset, vector or matrix
%
%   x = denormalize(x) Denormalize dataset x. x must be a structure created
%   by libDataset.create_dataset() and normalized by
%   libDataset.normalize().
%   
%   x = denormalize(x, mu_x, std_x) Denormalize vector or matrix x. If x is
%   a vector (it must be a column vector), mu_x and std_x must be a scalar.
%   If x is a matrix, mu_x and std_x must be a row vector with as many
%   columns as x.
%
    % Check input
    if nargin == 3
        % Denormalize
        x = bsxfun(@times, x, std_x); 
        x = bsxfun(@plus, x, mu_x);
    elseif nargin == 1
        % Check if dataset is legit and normalized
        libDataset.util_check_dataset(x, 'normalized');
        
        % Denormalize features
        x.x = libDataset.denormalize(x.x, x.norm_values.mu_x, x.norm_values.std_x);
        
    else
        error('You must provide normalization parameters.');
    end
    


end
