%% find_max_continuous.m
function [max_duration, max_start_idx] = find_max_continuous(binary_seq, dt)
    % Find maximum continuous true sequence
    % binary_seq: logical array
    % dt: time step in seconds
    
    if ~any(binary_seq)
        max_duration = 0;
        max_start_idx = 0;
        return;
    end
    
    % Find transitions
    diff_seq = diff([0; binary_seq(:); 0]);
    start_indices = find(diff_seq == 1);
    end_indices = find(diff_seq == -1) - 1;
    
    % Compute durations
    durations = (end_indices - start_indices + 1) * dt;
    
    % Find maximum
    [max_duration, max_idx] = max(durations);
    max_start_idx = start_indices(max_idx);
end