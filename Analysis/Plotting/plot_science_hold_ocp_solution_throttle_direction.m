function plot_science_hold_ocp_solution_throttle_direction(sol, opts)
% plot_science_hold_ocp_solution_throttle_direction
%
% Visualize science-hold OCP solution, including separated throttle and
% thrust-direction controls when present.
%
% Inputs:
%   sol  : solution struct from solve_science_hold_ocp
%   opts : optional struct
%          opts.timeUnit   = 'min' | 'hr' | 's'   (default 'min')
%          opts.lengthUnit = 'm'   | 'km'         (default 'm')
%          opts.show3D     = true/false           (default true)

if nargin < 2
    opts = struct();
end
if ~isfield(opts,'timeUnit'),   opts.timeUnit = 'min'; end
if ~isfield(opts,'lengthUnit'), opts.lengthUnit = 'm'; end
if ~isfield(opts,'show3D'),     opts.show3D = true; end

t = sol.t(:);
Nt = length(t);
Nc = sol.Ncollectors;
hasSeparatedControl = isfield(sol,'Alpha') && isfield(sol,'Dir');

% ---------------- Time scaling ----------------
switch lower(opts.timeUnit)
    case 's'
        tt = t;
        tlabel = 'Time [s]';
    case 'min'
        tt = t/60;
        tlabel = 'Time [min]';
    case 'hr'
        tt = t/3600;
        tlabel = 'Time [hr]';
    otherwise
        error('Unknown opts.timeUnit');
end

% ---------------- Length scaling ----------------
switch lower(opts.lengthUnit)
    case 'km'
        L = 1;
        llabel = 'km';
    case 'm'
        L = 1000;
        llabel = 'm';
    otherwise
        error('Unknown opts.lengthUnit');
end

% ---------------- OPD data ----------------
OPD = sol.OPD_km * L;                    % Nt x Nc
OPDspread = sol.OPDspread_km * L;        % Nt x 1
OPDmean = mean(OPD,2);
OPDrel = OPD - OPDmean;

% ---------------- Control norms ----------------
uNorm = zeros(Nt-1, Nc);
alphaHist = nan(Nt-1, Nc);
dirNorm = nan(Nt-1, Nc);
ttCtrl = tt(1:end-1);

for j = 1:Nc
    Uj = sol.U{j}; % (Nt-1) x 3
    uNorm(:,j) = sqrt(sum(Uj.^2,2));
    if hasSeparatedControl
        alphaHist(:,j) = sol.Alpha{j}(:);
        dirNorm(:,j) = sqrt(sum(sol.Dir{j}.^2,2));
    end
end

% ---------------- Mass ----------------
massHist = zeros(Nt, Nc);
for j = 1:Nc
    massHist(:,j) = sol.X{j}(:,7);
end

% ---------------- Figure 1: OPD ----------------
figure('Name','Science Hold OCP - OPD','Color','w');

subplot(3,1,1)
plot(tt, OPD, 'LineWidth',1.3)
xlabel(tlabel)
ylabel(['OPD [', llabel, ']'])
title('Per-Collector OPD')
grid on

subplot(3,1,2)
plot(tt, OPDrel, 'LineWidth',1.3)
xlabel(tlabel)
ylabel(['Relative OPD [', llabel, ']'])
title('OPD Relative to Mean')
grid on

subplot(3,1,3)
plot(tt, OPDspread, 'k', 'LineWidth',1.5); hold on
if isfield(sol,'Dmax_m')
    yline(sol.Dmax_m, 'r--', 'LineWidth',1.2);
end
xlabel(tlabel)
ylabel(['OPD spread [', llabel, ']'])
title('Formation OPD Spread')
grid on

%% ── OPD Spread – Journal-Quality Figure ──────────────────────────────────
figure('Units','inches','Position',[1 1 7 5]);

% Leave extra right margin so the D_max label never gets clipped
ax = axes('Units','normalized','Position',[0.10 0.11 0.80 0.81]);
hold(ax,'on');

% ── 0.  Unit / data setup ────────────────────────────────────────────────
unitFactor = 1;          % 1 → m | 0.01 → cm | 1e9 → nm
unitLabel  = 'm';
OPDplot    = OPDspread * unitFactor;

Dmax = 3;
if isfield(sol,'Dmax_m')
    Dmax = sol.Dmax_m * unitFactor;
end

t_settle = 10;           % minutes — edit to match your simulation
tMin     = tt(1);
tMax     = tt(end);
yTop     = max(OPDplot) * 1.18;   % 18 % headroom above data

% ── 2.  Green "feasible OPD" band (0 → Dmax, full time window) ───────────
if ~isnan(Dmax)
    patch(ax, [tMin tMax  tMax  tMin], ...
              [0    0     Dmax  Dmax], ...
        [0.18 0.72 0.18], ...
        'FaceAlpha',0.10, 'EdgeColor','none', 'HandleVisibility','off');
end

% ── 3.  Gray "settling / acquisition" band ───────────────────────────────
patch(ax, [tMin     t_settle  t_settle  tMin], ...
          [0        0         yTop      yTop], ...
    [0.6 0.6 0.6], ...
    'FaceAlpha',0.18, 'EdgeColor','none', 'HandleVisibility','off');

% ── 4.  Dmax horizontal dashed red line ──────────────────────────────────
if ~isnan(Dmax)
    yline(ax, Dmax, '--', ...
        'Color',     [0.85 0.125 0.098], ...
        'LineWidth',  1.4, ...
        'DisplayName', sprintf('$D_{\\max} = %.0f$ %s', Dmax, unitLabel));
end

% ── 5.  OPD spread (main curve) ──────────────────────────────────────────
plot(ax, tt, OPDplot, '-k', 'LineWidth', 1.8, ...
    'DisplayName', 'OPD spread $\max_j D_j - \min_j D_j$');

% ── 6.  Science-window statistics (mean + peak) ───────────────────────────
idx_sci                  = tt >= t_settle;
[T_sci, meanSci, peakSci, tPeakSci] = deal(NaN, NaN, NaN, NaN);

if any(idx_sci) && ~isnan(Dmax)
    OPD_sci  = OPDplot(idx_sci);
    tt_sci   = tt(idx_sci);
    meanSci  = mean(OPD_sci);
    T_sci    = tMax - t_settle;
    [peakSci, pkRel] = max(OPD_sci);
    tPeakSci = tt_sci(pkRel);

    yline(ax, meanSci, ':', ...
        'Color',     [0.0 0.447 0.741], ...
        'LineWidth',  1.2, ...
        'DisplayName', sprintf('Mean spread (sci) $\\approx %.1f$ %s', ...
                               meanSci, unitLabel));
end

% Settling boundary vertical line (no legend entry)
xline(ax, t_settle, '-', ...
    'Color',[0.4 0.4 0.4], 'LineWidth',1.0, 'HandleVisibility','off');

hold(ax,'off');

% ── 8.  Annotations ──────────────────────────────────────────────────────

% (a) "Settling / acquisition" — BOTTOM of gray band (avoids peak collision)
text(ax, tMin + 0.5*(t_settle - tMin), yTop * 0.05, ...
    {'Settling /'; 'acquisition'}, ...
    'Interpreter','latex', 'FontName','Times New Roman', ...
    'FontSize', 10, 'Color',[0.30 0.30 0.30], ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom');

% (b) "Science integration" label — upper-centre of science zone
if ~isnan(T_sci)
    text(ax, t_settle + 0.46*(tMax - t_settle), yTop * 0.93, ...
        sprintf('Science integration\\;\\;$T_{\\mathrm{sci}} = %.0f$ min', T_sci), ...
        'Interpreter','latex', 'FontName','Times New Roman', ...
        'FontSize', 10, 'Color',[0.10 0.45 0.10], ...
        'HorizontalAlignment','center', 'VerticalAlignment','top');
end

% (c) Peak-during-science annotation (avoids the settling zone entirely)
if ~isnan(peakSci)
    text(ax, tPeakSci, peakSci + yTop * 0.04, ...
        sprintf('Peak: $%.1f$ %s', peakSci, unitLabel), ...
        'Interpreter','latex', 'FontName','Times New Roman', ...
        'FontSize', 10, 'Color',[0 0 0], ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom');
end

% (d) Dmax label on right margin — 'Clipping','off' keeps it visible
if ~isnan(Dmax)
    text(ax, tMax + 0.4*(tMax-tMin)*0.01, Dmax, '$D_{\max}$', ...
        'Interpreter','latex', 'FontName','Times New Roman', ...
        'FontSize', 10, 'Color',[0.85 0.125 0.098], ...
        'HorizontalAlignment','left', 'VerticalAlignment','middle', ...
        'Clipping','off');
end

% ── 9.  Axis limits & labels ─────────────────────────────────────────────
ylim(ax, [0, yTop]);
xlim(ax, [tMin, tMax]);

xlabel(ax, tlabel,                                 'Interpreter','latex','FontSize',12);
ylabel(ax, sprintf('OPD spread [%s]', unitLabel),  'Interpreter','latex','FontSize',12);
title(ax,  'Formation OPD Spread',                 'Interpreter','latex','FontSize',13);

% ── 10.  Legend ───────────────────────────────────────────────────────────
lgd = legend(ax, 'Location','northeast', 'Interpreter','latex', 'Box','on');
lgd.FontSize = 10;

% ── 11.  Axes cosmetics ───────────────────────────────────────────────────
set(ax, ...
    'FontName',             'Times New Roman', ...
    'FontSize',              11, ...               % tick labels ≥ 10 pt
    'FontWeight',           'bold', ...
    'TickLabelInterpreter', 'latex', ...
    'LineWidth',             0.9, ...
    'Box',                  'on', ...
    'XMinorTick',           'on', ...
    'YMinorTick',           'on', ...
    'TickDir',              'in', ...
    'GridLineStyle',        '--', ...
    'GridAlpha',             0.3);

grid(ax,'on');
% ---------------- Figure 2: Controls ----------------
figure('Name','Science Hold OCP - Controls','Color','w');

if hasSeparatedControl
    for j = 1:Nc
        subplot(Nc,3,3*j-2)
        stairs(ttCtrl, sol.U{j}(:,1), 'LineWidth',1.1); hold on
        stairs(ttCtrl, sol.U{j}(:,2), 'LineWidth',1.1);
        stairs(ttCtrl, sol.U{j}(:,3), 'LineWidth',1.1);
        xlabel(tlabel)
        ylabel('u')
        title(sprintf('Collector %d equivalent RTN control', j))
        legend('u_R','u_T','u_N','Location','best')
        grid on

        subplot(Nc,3,3*j-1)
        stairs(ttCtrl, alphaHist(:,j), 'k', 'LineWidth',1.3)
        xlabel(tlabel)
        ylabel('\alpha')
        title(sprintf('Collector %d throttle', j))
        ylim([0, 1.05*max(1, max(alphaHist(:,j)))])
        grid on

        subplot(Nc,3,3*j)
        stairs(ttCtrl, sol.Dir{j}(:,1), 'LineWidth',1.1); hold on
        stairs(ttCtrl, sol.Dir{j}(:,2), 'LineWidth',1.1);
        stairs(ttCtrl, sol.Dir{j}(:,3), 'LineWidth',1.1);
        stairs(ttCtrl, dirNorm(:,j), 'k--', 'LineWidth',1.2);
        xlabel(tlabel)
        ylabel('d')
        title(sprintf('Collector %d thrust direction', j))
        legend('d_R','d_T','d_N','||d||','Location','best')
        grid on
    end
else
    for j = 1:Nc
        subplot(Nc,2,2*j-1)
        stairs(ttCtrl, sol.U{j}(:,1), 'LineWidth',1.1); hold on
        stairs(ttCtrl, sol.U{j}(:,2), 'LineWidth',1.1);
        stairs(ttCtrl, sol.U{j}(:,3), 'LineWidth',1.1);
        xlabel(tlabel)
        ylabel('u')
        title(sprintf('Collector %d RTN control', j))
        legend('u_R','u_T','u_N','Location','best')
        grid on

        subplot(Nc,2,2*j)
        stairs(ttCtrl, uNorm(:,j), 'k', 'LineWidth',1.3)
        xlabel(tlabel)
        ylabel('||u||')
        title(sprintf('Collector %d control norm', j))
        grid on
    end
end

% ---------------- Figure 3: Mass ----------------
figure('Name','Science Hold OCP - Mass','Color','w');
plot(tt, massHist, 'LineWidth',1.3)
xlabel(tlabel)
ylabel('Mass [kg]')
title('Collector Mass Histories')
grid on

% ---------------- Figure 4: RTN trajectories ----------------
if opts.show3D && isfield(sol,'RTN')
    figure('Name','Science Hold OCP - RTN 3D','Color','w');
    hold on
    grid on
    xlabel(['R [', llabel, ']'])
    ylabel(['T [', llabel, ']'])
    zlabel(['N [', llabel, ']'])
    title('Collector Trajectories in RTN')
    plot3(0,0,0,'ks','MarkerFaceColor','k','MarkerSize',8)

    cols = lines(Nc);

    if iscell(sol.RTN)
        for j = 1:Nc
            Rj = sol.RTN{j} * L;
            plot3(Rj(:,1), Rj(:,2), Rj(:,3), 'Color', cols(j,:), 'LineWidth',1.5)
            plot3(Rj(1,1), Rj(1,2), Rj(1,3), 'o', 'Color', cols(j,:), ...
                'MarkerFaceColor', cols(j,:), 'MarkerSize', 7)
            plot3(Rj(end,1), Rj(end,2), Rj(end,3), 's', 'Color', cols(j,:), ...
                'MarkerFaceColor', cols(j,:), 'MarkerSize', 7)
        end
    else
        % assume Nt x 3 x Nc
        for j = 1:Nc
            Rj = sol.RTN(:,:,j) * L;
            plot3(Rj(:,1), Rj(:,2), Rj(:,3), 'Color', cols(j,:), 'LineWidth',1.5)
            plot3(Rj(1,1), Rj(1,2), Rj(1,3), 'o', 'Color', cols(j,:), ...
                'MarkerFaceColor', cols(j,:), 'MarkerSize', 7)
            plot3(Rj(end,1), Rj(end,2), Rj(end,3), 's', 'Color', cols(j,:), ...
                'MarkerFaceColor', cols(j,:), 'MarkerSize', 7)
        end
    end
    axis equal
end

% ---------------- Figure 5: Pairwise separations ----------------
if isfield(sol,'pairSep_km')
    figure('Name','Science Hold OCP - Pairwise Separation','Color','w');
    plot(tt, sol.pairSep_km * L, 'LineWidth',1.3)
    xlabel(tlabel)
    ylabel(['Separation [', llabel, ']'])
    title('Pairwise Collector Separation')
    if isfield(sol,'pairSepLabels')
        legend(sol.pairSepLabels, 'Location','best')
    end
    grid on
end

% ---------------- Figure 6: Slack ----------------
if isfield(sol,'slackMax_km')
    figure('Name','Science Hold OCP - Slack','Color','w');
    plot(tt, sol.slackMax_km * L, 'k', 'LineWidth',1.4); hold on
    if isfield(sol,'slackPerCollector_km')
        plot(tt, sol.slackPerCollector_km * L, '--', 'LineWidth',1.1)
    end
    xlabel(tlabel)
    ylabel(['Slack [', llabel, ']'])
    title('OPD Constraint Slack')
    grid on
end

% ---------------- Console summary ----------------
fprintf('=============================================\n');
fprintf(' Science Hold OCP Solution Summary\n');
fprintf('=============================================\n');
fprintf('Success                 : %d\n', sol.success);
fprintf('Collectors              : %d\n', Nc);
fprintf('Duration                : %.3f s\n', t(end)-t(1));
fprintf('Max OPD spread          : %.6f %s\n', max(OPDspread), llabel);
if isfield(sol,'maxSlack_m')
    fprintf('Max OPD slack           : %.6f m\n', sol.maxSlack_m);
elseif isfield(sol,'maxSlack_km')
    fprintf('Max OPD slack           : %.6f m\n', 1000*sol.maxSlack_km);
end
if isfield(sol,'massUsed_kg')
    fprintf('Mass used per collector : '); fprintf('%.6f ', sol.massUsed_kg); fprintf('kg\n');
end
if isfield(sol,'massUsedTotal_kg')
    fprintf('Total mass used         : %.6f kg\n', sol.massUsedTotal_kg);
end
if hasSeparatedControl
    fprintf('Throttle range per coll.: ');
    for j = 1:Nc
        fprintf('[%.4f, %.4f] ', min(alphaHist(:,j)), max(alphaHist(:,j)));
    end
    fprintf('\n');
end
fprintf('=============================================\n');

end
