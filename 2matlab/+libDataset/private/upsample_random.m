function data = upsample_random(data, num_samples)
% Upsample dataset.sample_idx randomly, such that num_samples samples
% appear.
    data.sample_idx = datasample(data.sample_idx, num_samples);

end