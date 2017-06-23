function sets = split_random_subjects( data, split_param )
% Divide set into split_param.k sets. Subjects in one set will not appear
% in any other set. The subjects will be choosen randomly. 

    % Check dataset
    libDataset.util_check_dataset(data, 'subject');
    
    % Create output datasets
    sets = cell(split_param.k, 1);

    subject = data.subject(data.sample_idx);
    unique_subjects = unique(subject);
    num_subjects = size(unique_subjects, 1);
    num_samples = size(data.sample_idx, 1);
    
    % If the number of available subjects is less than the number of
    % desired sets, it's not possible to split
    if ~split_param.with_repetition && num_subjects < split_param.k
        error('Number of unique subjects is lower than the number of sets. Please reduce split_param.k.');
    end
    
    % Sample randomly subjects from data
    rand_subject = randperm(num_subjects);

    % Calculate the number of samples each subject provides
    subject_sample_idx = cell(num_subjects, 1);
    subject_num_samples = zeros(num_subjects, 1);
    for s = 1 : num_subjects
        subject_sample_idx{s} = data.sample_idx(subject == unique_subjects(s));
        subject_num_samples(s) = size(subject_sample_idx{s},1);
    end

    % Calculate the number of samples each set desires
    set_num_samples = zeros(split_param.k, 1);
    set_selected_subjects = cell(split_param.k, 1);
    for k = 1 : split_param.k
        set_num_samples(k) = floor(num_samples * split_param.ratio(k));
    end
    set_num_samples_left = set_num_samples;
    
    subject_counter = 1;
    set_idx = 1 : split_param.k;
    
    while ~isempty(set_idx) && subject_counter <= num_subjects
        
        % Prefer sets that have a lot of missing samples left
        [samples_left_sorted, set_idx] = sort(set_num_samples_left,'descend');
        
        % Do not sample sets that are already full
        set_idx(samples_left_sorted <= 0) = [];

        for k = set_idx'
            % Choose a random subject
            if split_param.with_repetition
                selected_subject = randi(num_subjects);
            else % without repetition
                selected_subject = rand_subject(subject_counter);
                % Each subject must be sampled only once
                subject_counter = subject_counter + 1;
            end
            
            % Add subject to selected list for current set
            set_selected_subjects{k} = [set_selected_subjects{k}; selected_subject];
            % Recalculate the number of samples that are still neccessary
            set_num_samples_left(k) = set_num_samples_left(k) - subject_num_samples(selected_subject);            
            
            % Break if no more subjects are available
            if subject_counter > num_subjects
                break;
            end
        end
    end
    
    set_num_samples_selected = set_num_samples - set_num_samples_left;
    
    % Finally sample each set according to the subjects that have been
    % addressed.
    for k = 1 : split_param.k
        if set_num_samples_selected(k) == 0
            warning(strcat('No samples selected for set k=', num2str(k), '.'));
        end
        
        sets{k} = data;
        sets{k}.sample_idx = zeros(set_num_samples_selected(k),1);
        % index for current set
        set_counter = 1;
        for set_subject = set_selected_subjects{k}'
            % Get index for selected subject
            sets{k}.sample_idx(set_counter : set_counter + subject_num_samples(set_subject) - 1) = subject_sample_idx{set_subject};
            % Prepare for next iteration
            set_counter = set_counter + subject_num_samples(set_subject);
        end
        
    end
end

