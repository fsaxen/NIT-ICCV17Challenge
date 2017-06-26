function [ model ] = train_rank_svm( feat, label, subj_id, emotion, ml_param, emo_in_featvec )
%train_rank_svm train rank SVM model

    
    X = NaN(size(feat));
    Y = NaN(size(label));
    
    % create samples for ranking by combining samples of the same
    % subject/emotion as pairs (subtract feature vectors)
    [~, ia, ic] = unique(horzcat(subj_id, emotion),'rows');
    for i = 1:length(ia)
        idx = ic == i;
        x_pair = feat(idx,:);
        y_pair = label(idx,:);
        X(idx,:) = vertcat(x_pair(1,:)-x_pair(2,:), x_pair(2,:)-x_pair(1,:));
        if y_pair(1) > y_pair(2)
            % x1 is (more) true (x2 is fake)
            Y(idx,:) = [ 1 ; 0 ];
        else
            % x2 is (more) true (x1 is fake)
            Y(idx,:) = [ 0 ; 1 ];
        end
    end
    
    % add emotion code to feature vector
    if emo_in_featvec
        eye6 = eye(6);
        emo_code = eye6(emotion,:);
        X = horzcat(X,emo_code);
    else
        emo_code = [];
    end

    % create and normalize dataset
    data_train = libDataset.create_dataset(X,Y,subj_id);
    data_train = libDataset.normalize(data_train);

    % train and save ranking/comparative model
    model = libML.train(data_train, ml_param);
    model.norm_values = data_train.norm_values;
    model.emo_in_featvec = emo_in_featvec;
    
    % train fallback SVM (that can handle single samples if no pairs are avaiable)
    X = horzcat(feat, emo_code);
    Y = label;
    data_train = libDataset.create_dataset(X,Y,subj_id);
    data_train = libDataset.normalize(data_train);
    model.fallback_model = libML.train(data_train, ml_param);
    model.fallback_model.norm_values = data_train.norm_values;
    
end

