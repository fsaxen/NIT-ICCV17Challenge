function [xvalues, yvalues, auc, threshold] = plot2D(p, gt, xmeasure, ymeasure, figure_id, names)
%PLOT2D Plots the roc or precision recall curves, specified by xmeasure and
%ymeasure.
%   For a number of possible measurements, please see libPerformance.vector

    % Import library to access private functions
    import libPerformance.*

    % Check input
    if any(size(p) ~= size(gt))
        error('p and gt must match in size.');
    end
    
    num_plots = size(p, 2);
    xvalues = zeros(size(p));
    yvalues = xvalues;
    threshold = xvalues;
    auc = zeros(1, num_plots);
    posx = zeros(1, num_plots);
    posy = zeros(1, num_plots);
    for i = 1 : num_plots
        % Calculate performances
        [xvalues(:,i), yvalues(:,i), ~, threshold(:,i)] = libPerformance.vector(p(:,i), gt(:,i), xmeasure, ymeasure, 'ConfusionMatrix', 'threshold');

        % Calculate area
        auc(i) = area_under_curve(xvalues(:,i), yvalues(:,i));
        
        % Calculate the current p > 0 position for x and y
        [~, idx0] = min(abs(threshold(:,i)));
        posx(i) = xvalues(idx0, i);
        posy(i) = yvalues(idx0, i);
    end
    
    % Plot
    if nargin >= 5 && isscalar(figure_id)
        figure(figure_id);
    else
        figure();
    end
    plot(xvalues, yvalues);
    ish = ishold;
    hold on
    plot(posx, posy, 'o');
    if ~ish
        hold off
    end
%     set(h2,'visible','off')
    xlabel(xmeasure); 
    ylabel(ymeasure); 
%     ylabel(h(2), 'threshold'); 
%     set(h(1),'box','on');
%     set(h(1),'YMinorTick', 'on');
%     set(h(1),'Ytick',linspace(0,1,6))
%     set(h(2),'Ytick',linspace(min(threshold),max(threshold),6))
    grid on;
%     grid minor;
    title(['AUC = ',num2str(auc)]);
    
    if nargin >= 6
        legend(names);
    end
end

