function parallelcoords_plus(X, C, var_names, alpha, X_highlight)

    [m,n] = size(X);
    
    if nargin < 2
        C = X(:,end);
    end
    if nargin < 3
        var_names = [];
    elseif length(var_names) ~= n && ~isempty(var_names)
        error('X and var_names must have equal number of columns.');
    end
    if nargin < 4
        alpha = max(min(1, 1-m/10000),0.01);
    end
    if nargin < 5
        X_highlight = [];
    end

    if length(C) ~= m
        error('X and C must have equal number of rows.');
    end
    
    %cmap = colormap(hsv(128));
    %cmap = colormap('cool');
    cmap = colormap;
    
    % figure out the range of the data
    rgs = [max(X,[],1); min(X,[],1)];
    nrm = rgs(1,:) - rgs(2,:) + eps;
    shft = rgs(2,:);
    
    % normalize data to [0 1]
    if range(C) > 0
        C_norm = (C - min(C)) ./ range(C);
    else
        C_norm = (C - min(C));
    end
    X = (X - ones(m,1)*shft)./ (ones(m,1)*nrm);

    % plot the data, one line per row
    not_nan = ~any(isnan(X),2) & ~isnan(C_norm(:));
    if ~any(not_nan)
        error('No non-NaN data!');
    end
    p = plot(1:n, X(not_nan,:)');
    
    % colorize
    caxis([min(C) max(C)+eps]);
    n_colors = size(cmap,1);
    colors_idx = round(C_norm(not_nan,:)*(n_colors-1)+1);
    colors = cmap(colors_idx,:);
    for i = 1:length(p)
        %set(p(i), 'Color', colors(i,:));
        p(i).Color = [colors(i,:) alpha];     % starting with R2014b
    end
    
    % highlight some data points
    hold on;
    for i = 1 : size(X_highlight,1)
        X_highlight(i,:) = (X_highlight(i,:) - shft) ./ nrm;
        plot(1:n, X_highlight(i,:),'ro');
    end
    
    % switch off the axes and draw own
    axis off;
    for i = 1:n
        % A line for every column
        line([i i],[0 1],'LineStyle','--','Color','k')
        % Show the lower and upper end of the respective y-axis
        text(i,-.01,num2str(rgs(2,i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
        text(i,1.01,num2str(rgs(1,i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    if ~isempty(var_names)
        for i = 1:n
            text(i,-.09, var_names{i}, 'Interpreter', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
        end
    end
    
    % Make the figure canvas white
    set(gcf,'Color','w')
    
    hold off;
    colorbar;
end