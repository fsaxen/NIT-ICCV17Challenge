function [ Y ] = predict_rank_svm( feat, subj_id, expr, model )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    emo_in_featvec = 1;
    % create features by combining samples of the same subject/emotion
    
    if emo_in_featvec
        X = NaN(size(feat,1),size(feat,2)+6);
    else
        X = NaN(size(feat));
    end    
    
    [~, ia, ic] = unique(horzcat(subj_id, expr),'rows');
    for i = 1:length(ia)
        idx = ic == i;
        x_cur = feat(idx,:);
        switch size(x_cur,1)
            case 1
                % TODO
            case 2
                if emo_in_featvec
                    X(idx,1:end-6) = vertcat(x_cur(1,:)-x_cur(2,:), x_cur(2,:)-x_cur(1,:));
                    emo_code = zeros(1,6);
                    emo_code(expr(find(idx,1))) = 1;
                    X(idx,end-5:end) = vertcat(emo_code,emo_code);
                else
                    X(idx,:) = vertcat(x_cur(1,:)-x_cur(2,:), x_cur(2,:)-x_cur(1,:));
                end

            otherwise
                % TODO
        end
    end

    data_test = libDataset.create_dataset(X);
    data_test = libDataset.normalize(data_test, model.norm_values);

    [y, p] = libML.predict(data_test, model);
        
    Y = NaN(size(p));
    for i = 1:length(ia)
        idx = ic == i;
        p_cur = p(idx,:);
        y_pair = y(idx,:);
        if p_cur(1) > p_cur(2)
            % x1 is (more) true (x2 is fake)
            Y(idx,:) = [ 1 ; 0 ];
        else
            % x2 is (more) true (x1 is fake)
            Y(idx,:) = [ 0 ; 1 ];
        end
    end

end

