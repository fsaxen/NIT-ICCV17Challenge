function [ pred ] = predict_rank_svm( feat, subj_id, emotion, model )
%predict_rank_svm predict with rank SVM model

    % create feature matrix (estimated number of rows)
    if model.emo_in_featvec
        X = NaN(size(feat,1),size(feat,2)+6);
    else
        X = NaN(size(feat));
    end
    pair_indices = NaN(size(feat,1),2);

    % for each original sample save the index range of (pair) samples
    pair_sample_index_range = NaN(size(feat,1),2);
    
    % prepare feature matrix (create pairs whenever applicable)
    [~, ia, ic] = unique(horzcat(subj_id, emotion),'rows');
    eye6 = eye(6);
    j = 1;
    for i = 1:length(ia)
        idx_orig = ic == i;
        feat_cur = feat(idx_orig,:);
        sample_indices = find(idx_orig)';
        switch length(sample_indices)
            case 1
                % only one subject: fallback to classic SVM (have 0 in pair indices)
                pair_indices_cur = [ 0 sample_indices(1) ];
                X_cur = feat_cur;
            case 2
                % default and expected case:
                % we compare two samples of same subject and emotion
                pair_indices_cur = [ sample_indices ; fliplr(sample_indices) ];
                X_cur = vertcat(feat_cur(1,:)-feat_cur(2,:), feat_cur(2,:)-feat_cur(1,:));
            otherwise
                % more than two samples of same subject and emotion to compare
                % (may be misclassified subject)
                % 1. create pairwise sample combinations indices
                [ii, jj] = meshgrid(sample_indices, sample_indices);
                % include both, (i,j) and (j,i), but exclude (i,i)
                ii = ii - diag(diag(ii));
                jj = jj - diag(diag(jj));
                pair_indices_cur = [ ii(ii > 0) jj(jj > 0) ];
                % 2. create feature vectors
                X_cur = feat(pair_indices_cur(:,1),:) - feat(pair_indices_cur(:,2),:);
        end
        % add emotion code
        if model.emo_in_featvec
            emo_code = eye6(emotion(sample_indices(1)),:);
            X_cur = horzcat(X_cur, repmat(emo_code,size(X_cur,1),1));
        end
        % gather everything
        k = j + size(X_cur,1) - 1;
        pair_indices(j:k,:) = pair_indices_cur;
        pair_sample_index_range(idx_orig,:) = repmat([j k],length(sample_indices),1);
        X(j:k,:) = X_cur;
        j = k + 1;
    end

    % prepare prediction variables
    idx_rank = pair_indices(:,1) ~= 0;
    y = NaN(size(idx_rank));
    p = NaN(size(idx_rank));
    
    % rank/comparative prediction of pairs
    data_test = libDataset.create_dataset(X(idx_rank,:));
    data_test = libDataset.normalize(data_test, model.norm_values);
    [y(idx_rank), p(idx_rank)] = libML.predict(data_test, model);
    
    % fallback prediction of single samples (only one samples of same subject/emotion available)
    if any(~idx_rank)
        data_test = libDataset.create_dataset(X(~idx_rank,:));
        data_test = libDataset.normalize(data_test, model.fallback_model.norm_values);
        [y(~idx_rank), p(~idx_rank)] = libML.predict(data_test, model.fallback_model);
    end
    
    % collect predictions of original input samples: fallback predictions first
    pred = NaN(size(feat,1),1);
    pred(pair_indices(~idx_rank,2)) = y(~idx_rank);
    
    % second, transfer pair prediction back to sample predictions
    for i = 1:length(ia)
        idx_orig = find(ic == i);
        idx_pairs = pair_sample_index_range(idx_orig(1),1) : pair_sample_index_range(idx_orig(1),2);
        if length(idx_pairs) == 1
            % skip fallback predictions
            continue;
        end
        % summarize scores of pairwise predictions for each original sample
        p_orig = NaN(1,length(idx_orig));
        for j = 1:length(p_orig)
            idx1 = find(pair_indices(:,1) == idx_orig(j));
            idx2 = find(pair_indices(:,2) == idx_orig(j));
            p_pair_cur = [p(idx1) ; -p(idx2)];
            p_orig(j) = mean(p_pair_cur);
        end
        % if scores are the same, prefer latter scores to get a
        % random split instead of a bias
        p_orig = p_orig + 1e-5 * (1:length(p_orig));
        % threshold by median
        pred_cur = p_orig > median(p_orig);
        pred(idx_orig) = pred_cur;
    end

end

