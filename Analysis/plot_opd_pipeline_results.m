function plot_opd_pipeline_results(t, out, varargin)
% plot_opd_pipeline_results
%
% Plot results from opd_pipeline_from_qns.
%
% Inputs:
%   t    : Nx1 time vector [s]
%   out  : output struct from opd_pipeline_from_qns
%
% Optional name-value pairs:
%   'TimeUnit'  : 'days' (default), 'hours', 'seconds'
%   'LengthUnit': 'm' (default), 'km'
%   'RateUnit'  : 'm/s' (default), 'km/s'
%   'MakeCumulativeOPD' : true (default) or false
%
% Plots:
%   1) ROE histories
%   2) ROE rate breakdown by perturbation
%   3) RTN relative position histories
%   4) OPD history
%   5) OPD rate breakdown
%   6) Cumulative OPD contribution by perturbation
%

% -------------------------------------------------------------------------
% Parse options
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'TimeUnit', 'days');
addParameter(p, 'LengthUnit', 'm');
addParameter(p, 'RateUnit', 'm/s');
addParameter(p, 'MakeCumulativeOPD', true);
parse(p, varargin{:});

timeUnit   = lower(p.Results.TimeUnit);
lengthUnit = lower(p.Results.LengthUnit);
rateUnit   = lower(p.Results.RateUnit);
makeCum    = p.Results.MakeCumulativeOPD;

% -------------------------------------------------------------------------
% Time scaling
% -------------------------------------------------------------------------
switch timeUnit
    case 'days'
        ts = t / 86400;
        txlab = 'Time (days)';
    case 'hours'
        ts = t / 3600;
        txlab = 'Time (hours)';
    case 'seconds'
        ts = t;
        txlab = 'Time (s)';
    otherwise
        error('Unknown TimeUnit.');
end

% -------------------------------------------------------------------------
% Length/rate scaling
% -------------------------------------------------------------------------
switch lengthUnit
    case 'm'
        L = 1000;      % km -> m
        llab = 'm';
    case 'km'
        L = 1;
        llab = 'km';
    otherwise
        error('Unknown LengthUnit.');
end

switch rateUnit
    case 'm/s'
        R = 1000;      % km/s -> m/s
        rlab = 'm/s';
    case 'km/s'
        R = 1;
        rlab = 'km/s';
    otherwise
        error('Unknown RateUnit.');
end

% -------------------------------------------------------------------------
% Extract data
% -------------------------------------------------------------------------
roe = out.roe;

roeDotTot  = out.roe_dot.total;
roeDotJ2   = out.roe_dot.J2;
roeDotSRP  = out.roe_dot.SRP;
roeDotMoon = out.roe_dot.Moon;
roeDotSun  = out.roe_dot.Sun;

dr_rtn = out.dr_rtn * L;     % convert to desired length unit
opd    = out.opd * L;        % convert to desired length unit

opdDotTot  = out.opd_dot.total * R;
opdDotJ2   = out.opd_dot.J2   * R;
opdDotSRP  = out.opd_dot.SRP  * R;
opdDotMoon = out.opd_dot.Moon * R;
opdDotSun  = out.opd_dot.Sun  * R;

% Labels
roeLabels = {'\delta a', '\delta\lambda', '\delta e_x', '\delta e_y', '\delta i_x', '\delta i_y'};
roeRateLabels = {'delta a dot', 'delta lambda dot', 'delta e_x dot', ...
                 'delta e_y dot', 'delta i_x dot', 'delta i_y dot'};
rtnLabels = {'\delta r_R', '\delta r_T', '\delta r_N'};

% -------------------------------------------------------------------------
% 1) ROE histories
% -------------------------------------------------------------------------
figure('Name','ROE Histories','Color','w');
for j = 1:6
    subplot(3,2,j)
    plot(ts, roe(:,j), 'k', 'LineWidth', 1.3);
    xlabel(txlab)
    ylabel(roeLabels{j}, 'Interpreter','tex')
    title(['ROE: ', roeLabels{j}], 'Interpreter','tex')
    grid on
end

% -------------------------------------------------------------------------
% 2) ROE rate breakdown by perturbation
% -------------------------------------------------------------------------
for j = 1:6
    figure('Name',['ROE Rate Breakdown ', roeRateLabels{j}],'Color','w');
    plot(ts, roeDotJ2(:,j),   'LineWidth',1.2); hold on
    plot(ts, roeDotSRP(:,j),  'LineWidth',1.2);
    plot(ts, roeDotMoon(:,j), 'LineWidth',1.2);
    plot(ts, roeDotSun(:,j),  'LineWidth',1.2);
    plot(ts, roeDotTot(:,j), 'k--', 'LineWidth',1.4);
    xlabel(txlab, 'Interpreter','tex')
    ylabel(roeRateLabels{j}, 'Interpreter','tex')
    title(['ROE Rate Breakdown: ', roeRateLabels{j}], 'Interpreter','tex')
    legend('J2','SRP','Moon','Sun grav','Total','Location','best')
    grid on
end

% -------------------------------------------------------------------------
% 3) RTN relative position histories
% -------------------------------------------------------------------------
figure('Name','RTN Relative Position','Color','w');
for j = 1:3
    subplot(3,1,j)
    plot(ts, dr_rtn(:,j), 'LineWidth',1.3);
    xlabel(txlab)
    ylabel([rtnLabels{j}, ' (', llab, ')'], 'Interpreter','tex')
    title(['RTN Relative Position: ', rtnLabels{j}], 'Interpreter','tex')
    grid on
end

% -------------------------------------------------------------------------
% 4) OPD history
% -------------------------------------------------------------------------
figure('Name','OPD History','Color','w');
plot(ts, opd, 'k', 'LineWidth',1.4);
xlabel(txlab)
ylabel(['OPD (', llab, ')'])
title('Optical Path Delay')
grid on

% -------------------------------------------------------------------------
% 5) OPD rate breakdown
% -------------------------------------------------------------------------
figure('Name','OPD Rate Breakdown','Color','w');
plot(ts, opdDotJ2,   'LineWidth',1.2); hold on
plot(ts, opdDotSRP,  'LineWidth',1.2);
plot(ts, opdDotMoon, 'LineWidth',1.2);
plot(ts, opdDotSun,  'LineWidth',1.2);
plot(ts, opdDotTot, 'k--', 'LineWidth',1.5);
xlabel(txlab)
ylabel(['OPD Rate (', rlab, ')'])
title('OPD Rate Breakdown by Perturbation')
legend('J2','SRP','Moon','Sun grav','Total','Location','best')
grid on

% -------------------------------------------------------------------------
% 6) Cumulative OPD contribution by perturbation
% -------------------------------------------------------------------------
if makeCum
    % Integrate OPD rate contributions over time
    opdCumJ2   = cumtrapz(t, opdDotJ2);
    opdCumSRP  = cumtrapz(t, opdDotSRP);
    opdCumMoon = cumtrapz(t, opdDotMoon);
    opdCumSun  = cumtrapz(t, opdDotSun);
    opdCumTot  = cumtrapz(t, opdDotTot);

    figure('Name','Cumulative OPD Contributions','Color','w');
    plot(ts, opdCumJ2,   'LineWidth',1.2); hold on
    plot(ts, opdCumSRP,  'LineWidth',1.2);
    plot(ts, opdCumMoon, 'LineWidth',1.2);
    plot(ts, opdCumSun,  'LineWidth',1.2);
    plot(ts, opdCumTot, 'k--', 'LineWidth',1.5);
    xlabel(txlab)
    ylabel(['Integrated OPD Contribution (', llab, ')'])
    title('Cumulative OPD Contribution by Perturbation')
    legend('J2','SRP','Moon','Sun grav','Total','Location','best')
    grid on
end

% -------------------------------------------------------------------------
% 7) Optional combined summary figure
% -------------------------------------------------------------------------
figure('Name','OPD Summary','Color','w');

subplot(2,2,1)
plot(ts, opd, 'k', 'LineWidth',1.3);
xlabel(txlab)
ylabel(['OPD (', llab, ')'])
title('OPD')
grid on

subplot(2,2,2)
plot(ts, opdDotJ2,   'LineWidth',1.1); hold on
plot(ts, opdDotSRP,  'LineWidth',1.1);
plot(ts, opdDotMoon, 'LineWidth',1.1);
plot(ts, opdDotSun,  'LineWidth',1.1);
plot(ts, opdDotTot, 'k--', 'LineWidth',1.4);
xlabel(txlab)
ylabel(['OPD Rate (', rlab, ')'])
title('OPD Rate Breakdown')
legend('J2','SRP','Moon','Sun grav','Total','Location','best')
grid on

subplot(2,2,3)
plot(ts, dr_rtn(:,1), 'LineWidth',1.1); hold on
plot(ts, dr_rtn(:,2), 'LineWidth',1.1);
plot(ts, dr_rtn(:,3), 'LineWidth',1.1);
xlabel(txlab)
ylabel(['RTN Position (', llab, ')'])
title('RTN Relative Position')
legend('\delta r_R','\delta r_T','\delta r_N','Location','best','Interpreter','tex')
grid on

subplot(2,2,4)
if makeCum
    plot(ts, opdCumJ2,   'LineWidth',1.1); hold on
    plot(ts, opdCumSRP,  'LineWidth',1.1);
    plot(ts, opdCumMoon, 'LineWidth',1.1);
    plot(ts, opdCumSun,  'LineWidth',1.1);
    plot(ts, opdCumTot, 'k--', 'LineWidth',1.4);
    xlabel(txlab)
    ylabel(['Integrated OPD (', llab, ')'])
    title('Cumulative OPD Contribution')
    legend('J2','SRP','Moon','Sun grav','Total','Location','best')
    grid on
else
    axis off
end

end