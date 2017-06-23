function [ model ] = train_rank_svm( feat, label, subj_id, expr, ml_param )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    emo_in_featvec = 1;
    % create features by combine samples of the same subject/emotion
    
    if emo_in_featvec
        X = NaN(size(feat,1),size(feat,2)+6);
    else
        X = NaN(size(feat));
    end
    Y = NaN(size(label));
    
    [~, ia, ic] = unique(horzcat(subj_id, expr),'rows');
    for i = 1:length(ia)
        idx = ic == i;
        x_pair = feat(idx,:);
        y_pair = label(idx,:);
        if emo_in_featvec
            X(idx,1:end-6) = vertcat(x_pair(1,:)-x_pair(2,:), x_pair(2,:)-x_pair(1,:));
            emo_code = zeros(1,6);
            emo_code(expr(find(idx,1))) = 1;
            X(idx,end-5:end) = vertcat(emo_code,emo_code);
        else
            X(idx,:) = vertcat(x_pair(1,:)-x_pair(2,:), x_pair(2,:)-x_pair(1,:));
        end
        if y_pair(1) > y_pair(2)
            % x1 is (more) true (x2 is fake)
            Y(idx,:) = [ 1 ; 0 ];
        else
            % x2 is (more) true (x1 is fake)
            Y(idx,:) = [ 0 ; 1 ];
        end
    end

    data_train = libDataset.create_dataset(X,Y,subj_id);
    data_train = libDataset.normalize(data_train);

    model = libML.train(data_train, ml_param);
    model.norm_values = data_train.norm_values;

end

