function figs = plot_qns_free_time_solution(sol, opts)
% plot_qns_free_time_solution
%
% Visualization utility for the single-deputy free-final-time QNS OCP
% solved by solve_qns_free_time_ocp.m.
%
% Usage:
%   figs = plot_qns_free_time_solution(sol)
%   figs = plot_qns_free_time_solution(sol, opts)
%
% Inputs
% ------
% sol  : solution struct returned by solve_qns_free_time_ocp
% opts : optional plotting settings struct
%
% Optional opts fields
% --------------------
% opts.timeUnit          : 'hours' (default), 'minutes', or 'seconds'
% opts.lineWidth         : default 1.8
% opts.fontSize          : default 11
% opts.showKeepOutSphere : default true
% opts.sphereAlpha       : default 0.08
% opts.make3D            : default true
% opts.makeOverview      : default true
% opts.makeElements      : default true
% opts.makeControls      : default true
% opts.makeStateHistory  : default true
% opts.makeSummaryText   : default true
%
% Output
% ------
% figs : struct of figure handles

validate_solution_struct(sol);
if nargin < 2, opts = struct(); end

opts = apply_plot_defaults(opts);

tSec = build_plot_time(sol);
[tPlot, timeLabel] = convert_time_axis(tSec, opts.timeUnit);

deltaR = sol.deltaR_km;
deltaAlpha = sol.deltaAlpha;
U = sol.U;
uNorm = sol.thrustNorm;
X = sol.X;

target = nan(1, 6);
if isfield(sol, 'deltaAlphaTarget') && ~isempty(sol.deltaAlphaTarget)
    target = sol.deltaAlphaTarget(:).';
elseif isfield(sol, 'problem') && isfield(sol.problem, 'deltaAlphaTarget')
    target = sol.problem.deltaAlphaTarget(:).';
end

dMin = nan;
if isfield(sol, 'problem') && isfield(sol.problem, 'dMin_km')
    dMin = sol.problem.dMin_km;
end

figs = struct();

if opts.makeOverview
    figs.overview = figure('Name', 'QNS Free-Time OCP Overview', 'Color', 'w');
    tl = tiledlayout(figs.overview, 3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Free-Final-Time QNS Retargeting Overview', 'FontWeight', 'bold');

    nexttile
    plot3(deltaR(:,1), deltaR(:,2), deltaR(:,3), 'LineWidth', opts.lineWidth, 'Color', [0.00 0.45 0.74]);
    hold on
    plot3(deltaR(1,1), deltaR(1,2), deltaR(1,3), 'o', 'MarkerFaceColor', [0.20 0.70 0.20], 'MarkerEdgeColor', 'k');
    plot3(deltaR(end,1), deltaR(end,2), deltaR(end,3), 's', 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k');
    if opts.showKeepOutSphere && isfinite(dMin) && dMin > 0
        draw_keep_out_sphere(dMin, opts.sphereAlpha);
        legend({'Trajectory', 'Start', 'Finish', 'Keep-out'}, 'Location', 'best')
    else
        legend({'Trajectory', 'Start', 'Finish'}, 'Location', 'best')
    end
    grid on
    axis equal
    view(35, 25)
    xlabel('\deltaR_R [km]')
    ylabel('\deltaR_T [km]')
    zlabel('\deltaR_N [km]')
    title('Relative RTN Geometry')

    nexttile
    plot(tPlot, sol.rangeToChief_km, 'LineWidth', opts.lineWidth, 'Color', [0.00 0.45 0.74]);
    hold on
    if isfinite(dMin)
        yline(dMin, '--', 'Min range', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
    end
    grid on
    xlabel(timeLabel)
    ylabel('Range to chief [km]')
    title('Chief Separation')

    nexttile
    plot(tPlot, X(:,7), 'LineWidth', opts.lineWidth, 'Color', [0.13 0.55 0.13]);
    hold on
    if isfield(sol, 'problem') && isfield(sol.problem, 'massDry_kg')
        yline(sol.problem.massDry_kg, '--', 'm_{dry}', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
    end
    grid on
    xlabel(timeLabel)
    ylabel('Mass [kg]')
    title('Mass History')

    nexttile
    plot(tPlot(1:end-1), U(:,1), 'LineWidth', opts.lineWidth, 'Color', [0.85 0.33 0.10]);
    hold on
    plot(tPlot(1:end-1), U(:,2), 'LineWidth', opts.lineWidth, 'Color', [0.00 0.45 0.74]);
    plot(tPlot(1:end-1), U(:,3), 'LineWidth', opts.lineWidth, 'Color', [0.47 0.67 0.19]);
    plot(tPlot(1:end-1), uNorm, '--', 'LineWidth', 1.5, 'Color', [0.25 0.25 0.25]);
    yline(1.0, ':', '||v|| = 1', 'LineWidth', 1.2, 'Color', [0.40 0.40 0.40]);
    grid on
    xlabel(timeLabel)
    ylabel('Control')
    title('Control History')
    legend({'v_R', 'v_T', 'v_N', '||v||'}, 'Location', 'best')

    nexttile
    if all(isfinite(target))
        err = sol.deltaAlphaFinal(:) - target(:);
        bar(err, 'FaceColor', [0.00 0.45 0.74], 'EdgeColor', 'none');
        hold on
        yline(0, 'k-');
        xticks(1:6)
        xticklabels({'\delta a', '\delta\lambda', '\delta e_x', '\delta e_y', '\delta i_x', '\delta i_y'})
        xtickangle(25)
        ylabel('Final error')
        title('Terminal Element Error')
        grid on
    else
        plot(tPlot, deltaAlpha, 'LineWidth', opts.lineWidth);
        grid on
        xlabel(timeLabel)
        ylabel('\delta\alpha')
        title('Relative Elements')
    end

    nexttile
    Jvals = collect_objective_terms(sol);
    bar(Jvals.values, 'FaceColor', [0.30 0.30 0.75], 'EdgeColor', 'none');
    xticks(1:numel(Jvals.labels))
    xticklabels(Jvals.labels)
    xtickangle(20)
    ylabel('Cost contribution')
    title('Objective Breakdown')
    grid on
end

if opts.make3D
    figs.geometry3d = figure('Name', 'QNS Relative Geometry', 'Color', 'w');
    tl = tiledlayout(figs.geometry3d, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Relative Motion in RTN', 'FontWeight', 'bold');

    nexttile([2 1])
    plot3(deltaR(:,1), deltaR(:,2), deltaR(:,3), 'LineWidth', 2.0, 'Color', [0.00 0.45 0.74]);
    hold on
    scatter3(0, 0, 0, 80, [0 0 0], 'filled')
    plot3(deltaR(1,1), deltaR(1,2), deltaR(1,3), 'o', 'MarkerFaceColor', [0.20 0.70 0.20], 'MarkerEdgeColor', 'k', 'MarkerSize', 7);
    plot3(deltaR(end,1), deltaR(end,2), deltaR(end,3), 's', 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k', 'MarkerSize', 7);
    if opts.showKeepOutSphere && isfinite(dMin) && dMin > 0
        draw_keep_out_sphere(dMin, opts.sphereAlpha);
        legend({'Deputy path', 'Chief', 'Start', 'Finish', 'Keep-out'}, 'Location', 'best')
    else
        legend({'Deputy path', 'Chief', 'Start', 'Finish'}, 'Location', 'best')
    end
    grid on
    axis equal
    view(40, 26)
    xlabel('\deltaR_R [km]')
    ylabel('\deltaR_T [km]')
    zlabel('\deltaR_N [km]')
    title('3D RTN Trajectory')

    nexttile
    plot(deltaR(:,1), deltaR(:,2), 'LineWidth', opts.lineWidth, 'Color', [0.85 0.33 0.10]);
    hold on
    plot(deltaR(1,1), deltaR(1,2), 'o', 'MarkerFaceColor', [0.20 0.70 0.20], 'MarkerEdgeColor', 'k');
    plot(deltaR(end,1), deltaR(end,2), 's', 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k');
    axis equal
    grid on
    xlabel('\deltaR_R [km]')
    ylabel('\deltaR_T [km]')
    title('R-T Projection')

    nexttile
    plot(deltaR(:,2), deltaR(:,3), 'LineWidth', opts.lineWidth, 'Color', [0.47 0.67 0.19]);
    hold on
    plot(deltaR(1,2), deltaR(1,3), 'o', 'MarkerFaceColor', [0.20 0.70 0.20], 'MarkerEdgeColor', 'k');
    plot(deltaR(end,2), deltaR(end,3), 's', 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k');
    axis equal
    grid on
    xlabel('\deltaR_T [km]')
    ylabel('\deltaR_N [km]')
    title('T-N Projection')
end

if opts.makeElements
    figs.elements = figure('Name', 'QNS Relative Elements', 'Color', 'w');
    tl = tiledlayout(figs.elements, 3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Relative Element Histories', 'FontWeight', 'bold');

    labels = {'\delta a', '\delta\lambda', '\delta e_x', '\delta e_y', '\delta i_x', '\delta i_y'};
    colors = lines(6);
    for k = 1:6
        nexttile
        plot(tPlot, deltaAlpha(:,k), 'LineWidth', opts.lineWidth, 'Color', colors(k,:));
        hold on
        if all(isfinite(target))
            yline(target(k), '--', 'Target', 'Color', [0.20 0.20 0.20], 'LineWidth', 1.2);
        end
        grid on
        xlabel(timeLabel)
        ylabel(labels{k})
        title(labels{k})
    end
end

if opts.makeControls
    figs.controls = figure('Name', 'QNS Controls', 'Color', 'w');
    tl = tiledlayout(figs.controls, 4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Control Components and Magnitude', 'FontWeight', 'bold');

    controlLabels = {'v_R', 'v_T', 'v_N'};
    controlColors = [0.85 0.33 0.10;
                     0.00 0.45 0.74;
                     0.47 0.67 0.19];

    for k = 1:3
        nexttile
        plot(tPlot(1:end-1), U(:,k), 'LineWidth', opts.lineWidth, 'Color', controlColors(k,:));
        grid on
        xlabel(timeLabel)
        ylabel(controlLabels{k})
        title([controlLabels{k} ' history'])
    end

    nexttile
    plot(tPlot(1:end-1), uNorm, 'LineWidth', opts.lineWidth, 'Color', [0.20 0.20 0.20]);
    hold on
    yline(1.0, '--', 'Control bound', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
    grid on
    xlabel(timeLabel)
    ylabel('||v||')
    title('Control Magnitude')
end

if opts.makeStateHistory
    figs.states = figure('Name', 'Deputy State History', 'Color', 'w');
    tl = tiledlayout(figs.states, 4, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Deputy QNS State History', 'FontWeight', 'bold');

    stateLabels = {'a [km]', 'e_x', 'e_y', 'i [rad]', '\Omega [rad]', 'u [rad]', 'm [kg]'};
    for k = 1:7
        nexttile
        plot(tPlot, X(:,k), 'LineWidth', opts.lineWidth, 'Color', [0.00 0.45 0.74]);
        grid on
        xlabel(timeLabel)
        ylabel(stateLabels{k})
        title(stateLabels{k})
    end

    nexttile
    axis off
    if opts.makeSummaryText
        summaryLines = build_summary_lines(sol);
        text(0.0, 1.0, summaryLines, 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'FontName', 'Consolas', 'FontSize', opts.fontSize);
    end
end

apply_font_size_to_figures(figs, opts.fontSize);
end

% =========================================================================
function validate_solution_struct(sol)
requiredFields = {'X', 'U', 'deltaR_km', 'deltaAlpha', 'thrustNorm'};
for k = 1:numel(requiredFields)
    if ~isfield(sol, requiredFields{k})
        error('plot_qns_free_time_solution:MissingField', ...
            'sol.%s is required for plotting.', requiredFields{k});
    end
end
end

% =========================================================================
function opts = apply_plot_defaults(opts)
if ~isfield(opts, 'timeUnit') || isempty(opts.timeUnit), opts.timeUnit = 'hours'; end
if ~isfield(opts, 'lineWidth') || isempty(opts.lineWidth), opts.lineWidth = 1.8; end
if ~isfield(opts, 'fontSize') || isempty(opts.fontSize), opts.fontSize = 11; end
if ~isfield(opts, 'showKeepOutSphere') || isempty(opts.showKeepOutSphere), opts.showKeepOutSphere = true; end
if ~isfield(opts, 'sphereAlpha') || isempty(opts.sphereAlpha), opts.sphereAlpha = 0.08; end
if ~isfield(opts, 'make3D') || isempty(opts.make3D), opts.make3D = true; end
if ~isfield(opts, 'makeOverview') || isempty(opts.makeOverview), opts.makeOverview = true; end
if ~isfield(opts, 'makeElements') || isempty(opts.makeElements), opts.makeElements = true; end
if ~isfield(opts, 'makeControls') || isempty(opts.makeControls), opts.makeControls = true; end
if ~isfield(opts, 'makeStateHistory') || isempty(opts.makeStateHistory), opts.makeStateHistory = true; end
if ~isfield(opts, 'makeSummaryText') || isempty(opts.makeSummaryText), opts.makeSummaryText = true; end
end

% =========================================================================
function tSec = build_plot_time(sol)
Nt = size(sol.X, 1);
if isfield(sol, 'h_s') && ~isempty(sol.h_s)
    tSec = (0:Nt-1).' * sol.h_s;
elseif isfield(sol, 'Tf_s') && ~isempty(sol.Tf_s)
    tSec = linspace(0, sol.Tf_s, Nt).';
elseif isfield(sol, 't') && ~isempty(sol.t)
    tSec = sol.t(:) - sol.t(1);
else
    tSec = (0:Nt-1).';
end
end

% =========================================================================
function [tPlot, label] = convert_time_axis(tSec, timeUnit)
switch lower(timeUnit)
    case 'seconds'
        tPlot = tSec;
        label = 'Time [s]';
    case 'minutes'
        tPlot = tSec / 60;
        label = 'Time [min]';
    otherwise
        tPlot = tSec / 3600;
        label = 'Time [hr]';
end
end

% =========================================================================
function draw_keep_out_sphere(radius, sphereAlpha)
[xs, ys, zs] = sphere(40);
surf(radius*xs, radius*ys, radius*zs, ...
    'FaceColor', [0.85 0.33 0.10], ...
    'FaceAlpha', sphereAlpha, ...
    'EdgeColor', 'none', ...
    'DisplayName', 'Keep-out');
end

% =========================================================================
function Jvals = collect_objective_terms(sol)
labels = {};
values = [];

if isfield(sol, 'Jtime'), labels{end+1} = 'Jtime'; values(end+1) = sol.Jtime; end %#ok<AGROW>
if isfield(sol, 'Jmass'), labels{end+1} = 'Jmass'; values(end+1) = sol.Jmass; end %#ok<AGROW>
if isfield(sol, 'Jctrl'), labels{end+1} = 'Jctrl'; values(end+1) = sol.Jctrl; end %#ok<AGROW>
if isfield(sol, 'Jterm'), labels{end+1} = 'Jterm'; values(end+1) = sol.Jterm; end %#ok<AGROW>

if isempty(values)
    labels = {'objective'};
    if isfield(sol, 'objective')
        values = sol.objective;
    else
        values = 0;
    end
end

Jvals.labels = labels;
Jvals.values = values;
end

% =========================================================================
function summaryLines = build_summary_lines(sol)
summaryLines = {};
summaryLines{end+1} = sprintf('success          : %d', logical(getfield_safe(sol, 'success', false))); %#ok<GFLD>
summaryLines{end+1} = sprintf('Tf [hr]          : %.6f', getfield_safe(sol, 'Tf_s', nan) / 3600); %#ok<GFLD>
summaryLines{end+1} = sprintf('mass used [kg]   : %.6f', getfield_safe(sol, 'massUsed_kg', nan)); %#ok<GFLD>
summaryLines{end+1} = sprintf('dv approx [m/s]  : %.6f', getfield_safe(sol, 'dvApprox_mps', nan)); %#ok<GFLD>
summaryLines{end+1} = sprintf('min range [km]   : %.6f', min(getfield_safe(sol, 'rangeToChief_km', nan))); %#ok<GFLD>

if isfield(sol, 'objective')
    summaryLines{end+1} = sprintf('objective        : %.6e', sol.objective);
end
if isfield(sol, 'Jtime')
    summaryLines{end+1} = sprintf('Jtime            : %.6e', sol.Jtime);
end
if isfield(sol, 'Jmass')
    summaryLines{end+1} = sprintf('Jmass            : %.6e', sol.Jmass);
end
if isfield(sol, 'Jctrl')
    summaryLines{end+1} = sprintf('Jctrl            : %.6e', sol.Jctrl);
end
if isfield(sol, 'Jterm')
    summaryLines{end+1} = sprintf('Jterm            : %.6e', sol.Jterm);
end

if isfield(sol, 'deltaAlphaError')
    err = sol.deltaAlphaError(:);
    labels = {'da', 'dlam', 'dex', 'dey', 'dix', 'diy'};
    for k = 1:min(6, numel(err))
        summaryLines{end+1} = sprintf('err %-12s: %.6e', labels{k}, err(k));
    end
end
end

% =========================================================================
function value = getfield_safe(S, fieldName, defaultValue)
if isstruct(S) && isfield(S, fieldName) && ~isempty(S.(fieldName))
    value = S.(fieldName);
else
    value = defaultValue;
end
end

% =========================================================================
function apply_font_size_to_figures(figs, fontSize)
figNames = fieldnames(figs);
for k = 1:numel(figNames)
    fig = figs.(figNames{k});
    if ishghandle(fig)
        set(findall(fig, 'Type', 'axes'), 'FontSize', fontSize);
    end
end
end
