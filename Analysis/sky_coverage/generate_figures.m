%% generate_figures.m
function generate_figures(RA_mesh, DEC_mesh, coverage_count, total_duration, ...
                         max_continuous, revisit_times, metrics, params, n_times)
    
    %% Figure 1: Multi-Panel Sky Coverage Map (PRIMARY FIGURE)
    figure('Position', [100, 100, 1400, 1000]);
    
    % Panel A: Coverage Percentage Map
    subplot(2, 2, 1);
    plot_sky_map(RA_mesh, DEC_mesh, metrics.coverage_percent, ...
                'Coverage Percentage (%)', [0, 100], 'turbo');
    title('(a) Observation Efficiency', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel B: Maximum Continuous Observation Time
    subplot(2, 2, 2);
    plot_sky_map(RA_mesh, DEC_mesh, metrics.max_continuous_hours, ...
                'Max Continuous (hours)', [0, prctile(metrics.max_continuous_hours(:), 95)], 'hot');
    title('(b) Maximum Continuous Observation', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel C: Total Observation Time
    subplot(2, 2, 3);
    plot_sky_map(RA_mesh, DEC_mesh, metrics.total_duration_hours, ...
                'Total Duration (hours)', [0, prctile(metrics.total_duration_hours(:), 95)], 'plasma');
    title('(c) Cumulative Observation Time', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel D: Sky Access (Binary)
    subplot(2, 2, 4);
    plot_sky_map(RA_mesh, DEC_mesh, double(metrics.ever_visible), ...
                'Accessible', [0, 1], 'gray');
    title('(d) Sky Accessibility Map', 'FontSize', 14, 'FontWeight', 'bold');
    
    sgtitle(sprintf('GEO Interferometer Sky Coverage Analysis (i = %.1f°, Year-Long Analysis)', ...
                   rad2deg(params.chief.i)), 'FontSize', 16, 'FontWeight', 'bold');
    
    if params.viz.save_figs
        saveas(gcf, fullfile(params.viz.fig_dir, 'fig1_sky_coverage_multipanel.png'));
        saveas(gcf, fullfile(params.viz.fig_dir, 'fig1_sky_coverage_multipanel.fig'));
    end
    
    %% Figure 2: Declination Band Analysis
    figure('Position', [150, 150, 800, 600]);
    bar(metrics.coverage_by_dec);
    set(gca, 'XTickLabel', metrics.dec_band_labels);
    xlabel('Declination Band', 'FontSize', 12);
    ylabel('Sky Coverage (%)', 'FontSize', 12);
    title('Sky Coverage by Declination Band', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0, 100]);
    
    if params.viz.save_figs
        saveas(gcf, fullfile(params.viz.fig_dir, 'fig2_declination_bands.png'));
    end
    
    %% Figure 3: Revisit Time Distribution
    if ~isempty(metrics.revisit_dist)
        figure('Position', [200, 200, 800, 600]);
        histogram(metrics.revisit_dist, 50, 'Normalization', 'probability');
        xlabel('Revisit Time (hours)', 'FontSize', 12);
        ylabel('Probability', 'FontSize', 12);
        title('Revisit Time Distribution', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % Add median line
        hold on;
        xline(metrics.median_revisit_hours, 'r--', 'LineWidth', 2, ...
              'Label', sprintf('Median: %.1f hrs', metrics.median_revisit_hours));
        hold off;
        
        if params.viz.save_figs
            saveas(gcf, fullfile(params.viz.fig_dir, 'fig3_revisit_distribution.png'));
        end
    end
    
    %% Figure 4: Observation Efficiency Histogram
    figure('Position', [250, 250, 800, 600]);
    histogram(metrics.coverage_percent(metrics.ever_visible), 50);
    xlabel('Observation Efficiency (%)', 'FontSize', 12);
    ylabel('Number of Sky Locations', 'FontSize', 12);
    title('Distribution of Observation Efficiency', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Add statistics
    text(0.6, 0.9, sprintf('Mean: %.1f%%\nMedian: %.1f%%', ...
         metrics.mean_efficiency, metrics.median_efficiency), ...
         'Units', 'normalized', 'FontSize', 11, 'BackgroundColor', 'white');
    
    if params.viz.save_figs
        saveas(gcf, fullfile(params.viz.fig_dir, 'fig4_efficiency_histogram.png'));
    end
    
    %% Figure 5: Science Targets Overlay (Example with Messier Objects)
    figure('Position', [300, 300, 1200, 700]);
    plot_sky_map(RA_mesh, DEC_mesh, metrics.coverage_percent, ...
                'Coverage (%)', [0, 100], 'turbo');
    
    % Overlay ecliptic and galactic planes
    hold on;
    
    % Ecliptic plane
    ra_ecliptic = 0:1:360;
    dec_ecliptic = 23.44 * sind(ra_ecliptic);  % Simplified
    plot(ra_ecliptic, dec_ecliptic, 'c--', 'LineWidth', 2, 'DisplayName', 'Ecliptic');
    
    % Galactic plane (very simplified)
    dec_galactic = -29 * ones(size(ra_ecliptic));
    plot(ra_ecliptic, dec_galactic, 'm--', 'LineWidth', 2, 'DisplayName', 'Galactic Plane');
    
    legend('Location', 'southeast');
    title('Sky Coverage with Celestial Reference Planes', 'FontSize', 14, 'FontWeight', 'bold');
    hold off;
    
    if params.viz.save_figs
        saveas(gcf, fullfile(params.viz.fig_dir, 'fig5_celestial_references.png'));
    end
    
end

%% Helper function for sky map plotting
function plot_sky_map(RA, DEC, data, clabel, climit, cmap_name)
    % Plot data on RA/Dec grid with proper formatting
    
    % Handle NaN and zero values for better visualization
    data_plot = data;
    data_plot(data_plot == 0 & climit(1) > 0) = NaN;
    
    % Create pseudocolor plot
    pcolor(RA, DEC, data_plot);
    shading interp;
    
    % Formatting
    xlabel('Right Ascension (deg)', 'FontSize', 11);
    ylabel('Declination (deg)', 'FontSize', 11);
    colormap jet
   cb = colorbar;
    cb.Label.String = clabel;
    cb.Label.FontSize = 11;
    clim(climit);
    
    % Grid and limits
    grid on;
    xlim([0, 360]);
    ylim([-90, 90]);
    set(gca, 'XTick', 0:60:360);
    set(gca, 'YTick', -90:30:90);
    axis equal tight;
    box on;
end