%% compute_coverage_metrics.m
function metrics = compute_coverage_metrics(coverage_count, total_duration, ...
                                           max_continuous, revisit_times, ...
                                           n_times, dt_analysis, RA_mesh, DEC_mesh)
    % Compute comprehensive coverage metrics
    
    % Total observation time available
    total_time_available = n_times * dt_analysis;
    
    %% Basic Coverage Statistics
    % Percentage of time each sky location is visible
    coverage_percent = 100 * coverage_count / n_times;
    
    % Total sky coverage (what fraction of sky is ever visible)
    ever_visible = coverage_count > 0;
    metrics.total_coverage_percent = 100 * sum(ever_visible(:)) / numel(ever_visible);
    
    % Mean observation efficiency across all targets
    metrics.mean_efficiency = mean(coverage_percent(ever_visible));
    metrics.median_efficiency = median(coverage_percent(ever_visible));
    
    %% Continuous Observation Statistics
    max_cont_hours = max_continuous / 3600;  % Convert to hours
    metrics.median_max_cont_hours = median(max_cont_hours(ever_visible));
    metrics.mean_max_cont_hours = mean(max_cont_hours(ever_visible));
    metrics.max_max_cont_hours = max(max_cont_hours(:));
    
    %% Revisit Time Statistics
    all_revisits = [];
    for i = 1:numel(revisit_times)
        if ~isempty(revisit_times{i})
            all_revisits = [all_revisits; revisit_times{i}(:)];
        end
    end
    
    if ~isempty(all_revisits)
        metrics.median_revisit_hours = median(all_revisits) / 3600;
        metrics.mean_revisit_hours = mean(all_revisits) / 3600;
        metrics.revisit_dist = all_revisits / 3600;  % Store for histogram
    else
        metrics.median_revisit_hours = NaN;
        metrics.mean_revisit_hours = NaN;
        metrics.revisit_dist = [];
    end
    
   %% Special Region Coverage (true ecliptic and galactic latitude bands)
    eps_obl = 23.4393;                                   % obliquity at J2000 [deg]
    v = [cosd(DEC_mesh(:)).*cosd(RA_mesh(:)), ...
         cosd(DEC_mesh(:)).*sind(RA_mesh(:)), ...
         sind(DEC_mesh(:))];                             % ICRS unit vectors
    lat_ecl = asind(-sind(eps_obl)*v(:,2) + cosd(eps_obl)*v(:,3));
    A_G3    = [-0.8676661490, -0.1980763734, 0.4559837762]; % third row, ICRS -> Galactic
    lat_gal = asind(v*A_G3.');
    ecliptic_mask = reshape(abs(lat_ecl) < 10, size(coverage_count));
    galactic_mask = reshape(abs(lat_gal) < 10, size(coverage_count));
    metrics.ecliptic_coverage       = 100*sum(ever_visible(ecliptic_mask))/sum(ecliptic_mask(:));
    metrics.galactic_coverage       = 100*sum(ever_visible(galactic_mask))/sum(galactic_mask(:));
    metrics.ecliptic_mean_percent   = mean(coverage_percent(ecliptic_mask));
    metrics.galactic_mean_percent   = mean(coverage_percent(galactic_mask));

    %% Additional statistics for the paper
    metrics.min_coverage_percent = min(coverage_percent(:));
    metrics.window_h = [1 2 3 4];
    metrics.pct_targets_with_window = arrayfun( ...
        @(w) 100*mean(max_continuous(:) >= w*3600), metrics.window_h);
    %% Coverage by Declination Band
    dec_bands = [-90, -60, -30, 0, 30, 60, 90];
    metrics.coverage_by_dec = zeros(length(dec_bands)-1, 1);
    for i = 1:length(dec_bands)-1
        band_mask = (DEC_mesh >= dec_bands(i)) & (DEC_mesh < dec_bands(i+1));
        metrics.coverage_by_dec(i) = 100 * sum(ever_visible(band_mask)) / sum(band_mask(:));
    end
    metrics.dec_band_labels = arrayfun(@(i) sprintf('%d to %d°', ...
                                       dec_bands(i), dec_bands(i+1)), ...
                                       1:length(dec_bands)-1, 'UniformOutput', false);
    
    %% Store processed maps for plotting
    metrics.coverage_percent = coverage_percent;
    metrics.max_continuous_hours = max_cont_hours;
    metrics.total_duration_hours = total_duration / 3600;
    metrics.ever_visible = ever_visible;
end