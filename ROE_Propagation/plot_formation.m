function ph = plot_formation(mode, axes_cell, data, ph_in)
%PLOT_FORMATION  Create or update all 8 formation flying visualization panels.
%
% Two operating modes:
%   'init'   - Creates all plot objects in the provided axes, returns handle struct.
%   'update' - Updates plot data in existing objects (fast, no axes clearing).
%
% Using 'update' mode is ~10x faster than 'init' for slider callbacks, since
% only XData/YData/ZData/CData are modified — no figure redraw overhead.
%
% Panel layout:
%   1: 3D RTN trajectory (time-colored: blue=early, red=late)
%   2: R-T plane projection (shows 2:1 CW ellipse)
%   3: T-N plane projection (interferometry perpendicularity plane)
%   4: R-N plane projection (passive safety — shows minimum separation)
%   5: ROE state time history (6 components, drift vs constant visible)
%   6: Eccentricity vector phase space (dex-dey plane)
%   7: Inclination vector phase space (dix-diy plane)
%   8: UV plane baseline coverage (interferometric aperture synthesis)
%
% Inputs:
%   mode       - 'init' (first call) or 'update' (subsequent calls)
%   axes_cell  - 1x8 cell array of axes handles {ax1,...,ax8}
%   data       - Struct with fields:
%                  .rtn       [3xN m] RTN positions
%                  .roe_traj  [6xN m] ROE time history
%                  .t_orbits  [1xN]   time in orbits
%                  .uv        [2xN m] UV coordinates (empty if disabled)
%                  .opd       [1xN m] optical path difference
%                  .metrics   struct from compute_metrics()
%                  .show_uv   logical
%   ph_in      - Handle struct from previous 'init' call (required for 'update')
%
% Outputs:
%   ph - Plot handle struct (returned from 'init'; passed through from 'update')

    switch lower(mode)
        case 'init'
            ph = local_init(axes_cell, data);
        case 'update'
            local_update(ph_in, data);
            ph = ph_in;
        otherwise
            error('plot_formation: mode must be ''init'' or ''update''');
    end
end

% =========================================================================
%  LOCAL FUNCTION: INIT — create all plot objects
% =========================================================================
function ph = local_init(ax, data)
    % Unpack data
    rtn       = data.rtn;
    roe_traj  = data.roe_traj;
    t_orbits  = data.t_orbits;
    N         = size(rtn, 2);
    t_color   = linspace(0, 1, N);   % 0=blue(start), 1=red(end) for colormap

    % Preallocate handle struct
    ph = struct();
    ph.ax = ax;

    % Color scheme
    roe_colors = lines(6);   % 6 distinct colors for ROE components

    % ---- Panel 1: 3D RTN trajectory ----
    ax1 = ax{1};
    colormap(ax1, cool);
    hold(ax1, 'on');
    ph.p1_traj = scatter3(ax1, rtn(1,:), rtn(2,:), rtn(3,:), ...
                          12, t_color, 'filled');
    ph.p1_orig = plot3(ax1, 0, 0, 0, 'k+', 'MarkerSize', 14, 'LineWidth', 2.5);
    ph.p1_start = plot3(ax1, rtn(1,1), rtn(2,1), rtn(3,1), ...
                        'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    ph.p1_end   = plot3(ax1, rtn(1,end), rtn(2,end), rtn(3,end), ...
                        'rs', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    grid(ax1, 'on'); box(ax1, 'on');
    xlabel(ax1, '\delta r_R [m]'); ylabel(ax1, '\delta r_T [m]');
    zlabel(ax1, '\delta r_N [m]');
    title(ax1, 'Relative Orbit — RTN Frame', 'FontWeight', 'bold');
    legend(ax1, {'Traj (t)', 'Chief', 'Start', 'End'}, 'Location', 'best', 'FontSize', 7);
    view(ax1, -35, 25);
    cb = colorbar(ax1); cb.Label.String = 'Normalized time'; cb.FontSize = 7;

    % ---- Panel 2: R-T plane projection ----
    ax2 = ax{2};
    colormap(ax2, cool);
    hold(ax2, 'on');
    ph.p2_traj = scatter(ax2, rtn(2,:), rtn(1,:), 8, t_color, 'filled');
    ph.p2_orig = plot(ax2, 0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
    grid(ax2, 'on'); axis(ax2, 'equal');
    xlabel(ax2, '\delta r_T [m]'); ylabel(ax2, '\delta r_R [m]');
    title(ax2, 'R-T Plane (2:1 CW Ellipse)', 'FontWeight', 'bold');
    ax2.XDir = 'normal';  ax2.YDir = 'normal';

    % ---- Panel 3: T-N plane projection ----
    ax3 = ax{3};
    colormap(ax3, cool);
    hold(ax3, 'on');
    ph.p3_traj = scatter(ax3, rtn(2,:), rtn(3,:), 8, t_color, 'filled');
    ph.p3_orig = plot(ax3, 0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
    % Star direction line (if UV enabled and star has T/N components)
    ph.p3_star = plot(ax3, [0 0], [0 0], 'm--', 'LineWidth', 1.5, 'Visible', 'off');
    grid(ax3, 'on'); axis(ax3, 'equal');
    xlabel(ax3, '\delta r_T [m]'); ylabel(ax3, '\delta r_N [m]');
    title(ax3, 'T-N Plane (Interferometry Baseline)', 'FontWeight', 'bold');

    % ---- Panel 4: R-N plane projection ----
    ax4 = ax{4};
    colormap(ax4, cool);
    hold(ax4, 'on');
    ph.p4_traj = scatter(ax4, rtn(1,:), rtn(3,:), 8, t_color, 'filled');
    ph.p4_orig = plot(ax4, 0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
    % Minimum separation circle
    theta_c = linspace(0, 2*pi, 200);
    rmin    = data.metrics.rn_min_sep;
    ph.p4_circle = plot(ax4, rmin*cos(theta_c), rmin*sin(theta_c), ...
                        'r--', 'LineWidth', 1.5);
    grid(ax4, 'on'); axis(ax4, 'equal');
    xlabel(ax4, '\delta r_R [m]'); ylabel(ax4, '\delta r_N [m]');
    title(ax4, 'R-N Plane (Passive Safety)', 'FontWeight', 'bold');
    ph.p4_title_txt = text(ax4, 0.02, 0.97, ...
        sprintf('R-N_{min} = %.1f m', rmin), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8, 'Color', 'r');

    % ---- Panel 5: ROE time history ----
    ax5 = ax{5};
    hold(ax5, 'on');
    roe_labels = {'a\cdot\deltaa', 'a\cdot\delta\lambda', ...
                  'a\cdot\deltae_x', 'a\cdot\deltae_y', ...
                  'a\cdot\deltai_x', 'a\cdot\deltai_y'};
    ph.p5_lines = gobjects(6, 1);
    for k = 1:6
        ph.p5_lines(k) = plot(ax5, t_orbits, roe_traj(k,:), ...
                              'Color', roe_colors(k,:), 'LineWidth', 1.5);
    end
    ph.p5_zero = plot(ax5, [t_orbits(1) t_orbits(end)], [0 0], ...
                      'k:', 'LineWidth', 0.8);
    grid(ax5, 'on');
    xlabel(ax5, 'Time [orbits]'); ylabel(ax5, 'ROE [m]');
    title(ax5, 'ROE State Time History', 'FontWeight', 'bold');
    legend(ax5, roe_labels, 'Location', 'best', 'FontSize', 7, 'Interpreter', 'tex');

    % ---- Panel 6: Eccentricity vector phase space ----
    ax6 = ax{6};
    hold(ax6, 'on');
    dex0 = roe_traj(3,1); dey0 = roe_traj(4,1);
    dix0 = roe_traj(5,1); diy0 = roe_traj(6,1);
    % de vector arrow from origin
    ph.p6_de_vec = quiver(ax6, 0, 0, dex0, dey0, 0, ...
                          'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    ph.p6_de_pt  = plot(ax6, dex0, dey0, 'bs', 'MarkerSize', 8, ...
                        'MarkerFaceColor', 'b');
    % di vector projected onto e-plane (for e/i angle reference)
    ph.p6_di_vec = quiver(ax6, 0, 0, dix0, diy0, 0, ...
                          'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5, ...
                          'LineStyle', '--');
    ph.p6_di_pt  = plot(ax6, dix0, diy0, 'r^', 'MarkerSize', 8, ...
                        'MarkerFaceColor', 'r');
    plot(ax6, 0, 0, 'k.', 'MarkerSize', 12);
    grid(ax6, 'on'); axis(ax6, 'equal');
    xlabel(ax6, 'a\cdot\deltae_x [m]'); ylabel(ax6, 'a\cdot\deltae_y [m]');
    title(ax6, 'Eccentricity Vector Phase Space', 'FontWeight', 'bold');
    legend(ax6, {'\delta\bfe (blue)', '\delta\bfi (red, proj.)'}, ...
           'FontSize', 7, 'Location', 'best');
    ph.p6_angle_txt = text(ax6, 0.02, 0.97, '', ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8);

    % ---- Panel 7: Inclination vector phase space ----
    ax7 = ax{7};
    hold(ax7, 'on');
    ph.p7_di_vec = quiver(ax7, 0, 0, dix0, diy0, 0, ...
                          'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    ph.p7_di_pt  = plot(ax7, dix0, diy0, 'r^', 'MarkerSize', 8, ...
                        'MarkerFaceColor', 'r');
    ph.p7_de_vec = quiver(ax7, 0, 0, dex0, dey0, 0, ...
                          'b', 'LineWidth', 1.5, 'MaxHeadSize', 0.5, ...
                          'LineStyle', '--');
    ph.p7_de_pt  = plot(ax7, dex0, dey0, 'bs', 'MarkerSize', 8, ...
                        'MarkerFaceColor', 'b');
    plot(ax7, 0, 0, 'k.', 'MarkerSize', 12);
    grid(ax7, 'on'); axis(ax7, 'equal');
    xlabel(ax7, 'a\cdot\deltai_x [m]'); ylabel(ax7, 'a\cdot\deltai_y [m]');
    title(ax7, 'Inclination Vector Phase Space', 'FontWeight', 'bold');
    legend(ax7, {'\delta\bfi (red)', '\delta\bfe (blue, proj.)'}, ...
           'FontSize', 7, 'Location', 'best');

    % ---- Panel 8: UV plane coverage ----
    ax8 = ax{8};
    colormap(ax8, cool);
    hold(ax8, 'on');
    if data.show_uv && ~isempty(data.uv)
        uv = data.uv;
        ph.p8_uv = scatter(ax8, uv(1,:), uv(2,:), 8, t_color, 'filled');
        ph.p8_orig = plot(ax8, 0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
        ph.p8_vis = true;
    else
        ph.p8_uv   = scatter(ax8, 0, 0, 8, 0.5, 'filled');
        ph.p8_orig = plot(ax8, 0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
        ph.p8_vis  = false;
    end
    grid(ax8, 'on'); axis(ax8, 'equal');
    xlabel(ax8, 'u [m]'); ylabel(ax8, 'v [m]');
    title(ax8, 'UV Plane (Interferometric Baseline)', 'FontWeight', 'bold');
    if ~data.show_uv
        text(ax8, 0.5, 0.5, 'UV panel disabled', 'Units', 'normalized', ...
             'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
    end

    % Store data snapshot for limits computation
    ph.last_data = data;
end

% =========================================================================
%  LOCAL FUNCTION: UPDATE — modify existing plot objects (fast path)
% =========================================================================
function local_update(ph, data)
    rtn      = data.rtn;
    roe_traj = data.roe_traj;
    t_orbits = data.t_orbits;
    N        = size(rtn, 2);
    t_color  = linspace(0, 1, N);

    % ---- Panel 1: 3D RTN ----
    set(ph.p1_traj,  'XData', rtn(1,:), 'YData', rtn(2,:), 'ZData', rtn(3,:), ...
                     'CData', t_color);
    set(ph.p1_start, 'XData', rtn(1,1),   'YData', rtn(2,1),   'ZData', rtn(3,1));
    set(ph.p1_end,   'XData', rtn(1,end), 'YData', rtn(2,end), 'ZData', rtn(3,end));

    % ---- Panel 2: R-T plane ----
    set(ph.p2_traj, 'XData', rtn(2,:), 'YData', rtn(1,:), 'CData', t_color);
    ax2 = ph.ax{2};
    axis(ax2, 'auto');
    % Ensure a minimum visible range so single-point trajectories are legible
    xl = xlim(ax2); yl = ylim(ax2);
    if diff(xl) < 20, xlim(ax2, mean(xl) + [-10 10]); end
    if diff(yl) < 20, ylim(ax2, mean(yl) + [-10 10]); end
    axis(ax2, 'equal');

    % ---- Panel 3: T-N plane ----
    set(ph.p3_traj, 'XData', rtn(2,:), 'YData', rtn(3,:), 'CData', t_color);
    ax3 = ph.ax{3};
    axis(ax3, 'auto');
    xl = xlim(ax3); yl = ylim(ax3);
    if diff(xl) < 20, xlim(ax3, mean(xl) + [-10 10]); end
    if diff(yl) < 20, ylim(ax3, mean(yl) + [-10 10]); end
    axis(ax3, 'equal');

    % ---- Panel 4: R-N plane + min-sep circle ----
    set(ph.p4_traj, 'XData', rtn(1,:), 'YData', rtn(3,:), 'CData', t_color);
    rmin = data.metrics.rn_min_sep;
    theta_c = linspace(0, 2*pi, 200);
    set(ph.p4_circle, 'XData', rmin*cos(theta_c), 'YData', rmin*sin(theta_c));
    set(ph.p4_title_txt, 'String', sprintf('R-N_{min} = %.1f m', rmin));
    ax4 = ph.ax{4}; axis(ax4, 'auto'); axis(ax4, 'equal');

    % ---- Panel 5: ROE time history ----
    for k = 1:6
        set(ph.p5_lines(k), 'XData', t_orbits, 'YData', roe_traj(k,:));
    end
    set(ph.p5_zero, 'XData', [t_orbits(1) t_orbits(end)], 'YData', [0 0]);
    ax5 = ph.ax{5};
    xl = [t_orbits(1) t_orbits(end)];
    xlim(ax5, xl + [-0.05 0.05]*diff(xl));

    % ---- Panel 6: Eccentricity phase space ----
    dex0 = roe_traj(3,1); dey0 = roe_traj(4,1);
    dix0 = roe_traj(5,1); diy0 = roe_traj(6,1);
    set(ph.p6_de_vec, 'UData', dex0, 'VData', dey0);
    set(ph.p6_de_pt,  'XData', dex0, 'YData', dey0);
    set(ph.p6_di_vec, 'UData', dix0, 'VData', diy0);
    set(ph.p6_di_pt,  'XData', dix0, 'YData', diy0);
    if ~isnan(data.metrics.angle_de_di)
        set(ph.p6_angle_txt, 'String', ...
            sprintf('|\\deltae|=%.0fm, \\angle(e,i)=%.0f°', ...
                    data.metrics.de_norm, data.metrics.angle_de_di));
    else
        set(ph.p6_angle_txt, 'String', ...
            sprintf('|\\deltae|=%.0fm', data.metrics.de_norm));
    end
    ax6 = ph.ax{6}; axis(ax6, 'auto'); axis(ax6, 'equal');

    % ---- Panel 7: Inclination phase space ----
    set(ph.p7_di_vec, 'UData', dix0, 'VData', diy0);
    set(ph.p7_di_pt,  'XData', dix0, 'YData', diy0);
    set(ph.p7_de_vec, 'UData', dex0, 'VData', dey0);
    set(ph.p7_de_pt,  'XData', dex0, 'YData', dey0);
    ax7 = ph.ax{7}; axis(ax7, 'auto'); axis(ax7, 'equal');

    % ---- Panel 8: UV plane ----
    if data.show_uv && ~isempty(data.uv)
        uv = data.uv;
        set(ph.p8_uv, 'XData', uv(1,:), 'YData', uv(2,:), 'CData', t_color);
        ax8 = ph.ax{8}; axis(ax8, 'auto'); axis(ax8, 'equal');
    end

    ph.last_data = data;
end
