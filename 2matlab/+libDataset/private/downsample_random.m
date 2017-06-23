function data = downsample_random(data, num_samples)
% Simply sample randomly with num_samples samples.
    
    if num_samples > length(data.sample_idx)
        error('This is a downsampling technique. You ask to provide more samples than available.');
    end
    
    data.sample_idx = data.sample_idx(randperm(length(data.sample_idx), num_samples));

end