function area = area_under_curve(x, y)
%AUC calcules the area under curve 
%   For e.g. area under ROC or area under recall precision

    % Check input
    if any(size(x) ~= size(y))
        error('x and y must have the same number of elements');
    end
    if isempty(x)
        error('Empty input vectors.');
    end
    
    N = size(x, 1);
    
    % sort x in ascending order to assure that we can start from 0
    [xs, idx] = sort(x, 'ascend');
    ys = y(idx,:);
    
    % First element
    area = xs(1) * ys(1);
    for i = 2 : N
        area = area + (xs(i) - xs(i-1)) * (ys(i) + ys(i-1));
    end
    area = area / 2;
end

