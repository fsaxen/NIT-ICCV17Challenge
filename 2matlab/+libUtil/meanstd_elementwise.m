function [ mat_mean, mat_std ] = meanstd_elementwise( mat_cell_array )
%MEANSTD_ELEMENTWISE Calculate element-wise mean and standard deviation
%   from a cell array of equally sized matrices

    if ~iscell(mat_cell_array)
        error('Argument must be a cell array of equally sized matrices!');
    end
    n = length(mat_cell_array);
    
    mat_sum = double(mat_cell_array{1});
    for i = 2:n
        mat_sum = mat_sum + double(mat_cell_array{i});
    end
    mat_mean = mat_sum ./ n;
    
    if nargout > 1
        mat_sum = zeros(size(mat_mean));
        for i = 1:n
            mat_sum = mat_sum + (double(mat_cell_array{i}) - mat_mean) .^ 2;
        end
        mat_std = sqrt(mat_sum ./ (n-1));
    end

end

