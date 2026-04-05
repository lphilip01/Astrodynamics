function plot_opd_pipeline_results(t, out, varargin)
% plot_opd_pipeline_results
%
% Plot results from opd_pipeline_from_qns.
% Handles:
%   - single deputy output (same as before)
%   - multi-deputy output (cell arrays)
%

% Detect multi-deputy
if iscell(out.roe)
    nDep = numel(out.roe);
    for j = 1:nDep
        subout = local_extract_single_out(out, j);
        plot_opd_pipeline_results_single(t, subout, varargin{:}, 'DeputyIndex', j);
    end
else
    plot_opd_pipeline_results_single(t, out, varargin{:});
end

end

% =========================================================================
function plot_opd_pipeline_results_single(t, out, varargin)

p = inputParser;
addParameter(p, 'TimeUnit', 'days');
addParameter(p, 'LengthUnit', 'm');
addParameter(p, 'RateUnit', 'm/s');
addParameter(p, 'MakeCumulativeOPD', true);
addParameter(p, 'DeputyIndex', []);
parse(p, varargin{:});

timeUnit   = lower(p.Results.TimeUnit);
lengthUnit = lower(p.Results.LengthUnit);
rateUnit   = lower(p.Results.RateUnit);
makeCum    = p.Results.MakeCumulativeOPD;
depIdx     = p.Results.DeputyIndex;

switch timeUnit
    case 'days'
        ts = t / 86400; txlab = 'Time (days)';
    case 'hours'
        ts = t / 3600; txlab = 'Time (hours)';
    case 'seconds'
        ts = t; txlab = 'Time (s)';
    otherwise
        error('Unknown TimeUnit.');
end

switch lengthUnit
    case 'm'
        L = 1000; llab = 'm';
    case 'km'
        L = 1; llab = 'km';
    otherwise
        error('Unknown LengthUnit.');
end

switch rateUnit
    case 'm/s'
        R = 1000; rlab = 'm/s';
    case 'km/s'
        R = 1; rlab = 'km/s';
    otherwise
        error('Unknown RateUnit.');
end

tag = '';
if ~isempty(depIdx)
    tag = sprintf(' (Deputy %d)', depIdx);
end

roe = out.roe;
roeDotTot  = out.roe_dot.total;
roeDotJ2   = out.roe_dot.J2;
roeDotSRP  = out.roe_dot.SRP;
roeDotMoon = out.roe_dot.Moon;
roeDotSun  = out.roe_dot.Sun;

dr_rtn = out.dr_rtn * L;
opd    = out.opd * L;

opdDotTot  = out.opd_dot.total * R;
opdDotJ2   = out.opd_dot.J2   * R;
opdDotSRP  = out.opd_dot.SRP  * R;
opdDotMoon = out.opd_dot.Moon * R;
opdDotSun  = out.opd_dot.Sun  * R;

roeLabels = {'\delta a', '\delta\lambda', '\delta e_x', '\delta e_y', '\delta i_x', '\delta i_y'};
roeRateLabels = {'delta a dot', 'delta lambda dot', 'delta e_x dot', ...
                 'delta e_y dot', 'delta i_x dot', 'delta i_y dot'};
rtnLabels = {'\delta r_R', '\delta r_T', '\delta r_N'};

figure('Name',['ROE Histories', tag],'Color','w');
for j = 1:6
    subplot(3,2,j)
    plot(ts, roe(:,j), 'k', 'LineWidth', 1.3);
    xlabel(txlab)
    ylabel(roeLabels{j}, 'Interpreter','tex')
    title(['ROE: ', roeLabels{j}, tag], 'Interpreter','tex')
    grid on
end

% for j = 1:6
%     figure('Name',[roeRateLabels{j}, tag],'Color','w');
%     plot(ts, roeDotJ2(:,j),   'LineWidth',1.2); hold on
%     plot(ts, roeDotSRP(:,j),  'LineWidth',1.2);
%     plot(ts, roeDotMoon(:,j), 'LineWidth',1.2);
%     plot(ts, roeDotSun(:,j),  'LineWidth',1.2);
%     plot(ts, roeDotTot(:,j), 'k--', 'LineWidth',1.4);
%     xlabel(txlab)
%     ylabel(roeRateLabels{j})
%     title(['ROE Rate Breakdown: ', roeRateLabels{j}, tag])
%     legend('J2','SRP','Moon','Sun grav','Total','Location','best')
%     grid on
% end

% figure('Name',['RTN Relative Position', tag],'Color','w');
% for j = 1:3
%     subplot(3,1,j)
%     plot(ts, dr_rtn(:,j), 'LineWidth',1.3);
%     xlabel(txlab)
%     ylabel([rtnLabels{j}, ' (', llab, ')'], 'Interpreter','tex')
%     title(['RTN Relative Position: ', rtnLabels{j}, tag], 'Interpreter','tex')
%     grid on
% end

% figure('Name',['OPD History', tag],'Color','w');
% plot(ts, opd, 'k', 'LineWidth',1.4);
% xlabel(txlab)
% ylabel(['OPD (', llab, ')'])
% title(['Optical Path Delay', tag])
% grid on
% 
% figure('Name',['OPD Rate Breakdown', tag],'Color','w');
% plot(ts, opdDotJ2,   'LineWidth',1.2); hold on
% plot(ts, opdDotSRP,  'LineWidth',1.2);
% plot(ts, opdDotMoon, 'LineWidth',1.2);
% plot(ts, opdDotSun,  'LineWidth',1.2);
% plot(ts, opdDotTot, 'k--', 'LineWidth',1.5);
% xlabel(txlab)
% ylabel(['OPD Rate (', rlab, ')'])
% title(['OPD Rate Breakdown by Perturbation', tag])
% legend('J2','SRP','Moon','Sun grav','Total','Location','best')
% grid on

if makeCum
    opdCumJ2   = cumtrapz(t, opdDotJ2);
    opdCumSRP  = cumtrapz(t, opdDotSRP);
    opdCumMoon = cumtrapz(t, opdDotMoon);
    opdCumSun  = cumtrapz(t, opdDotSun);
    opdCumTot  = cumtrapz(t, opdDotTot);

    figure('Name',['Cumulative OPD Contributions', tag],'Color','w');
    plot(ts, opdCumJ2,   'LineWidth',1.2); hold on
    plot(ts, opdCumSRP,  'LineWidth',1.2);
    plot(ts, opdCumMoon, 'LineWidth',1.2);
    plot(ts, opdCumSun,  'LineWidth',1.2);
    plot(ts, opdCumTot, 'k--', 'LineWidth',1.5);
    xlabel(txlab)
    ylabel(['Integrated OPD Contribution (', llab, ')'])
    title(['Cumulative OPD Contribution by Perturbation', tag])
    legend('J2','SRP','Moon','Sun grav','Total','Location','best')
    grid on
end

figure('Name',['OPD Summary', tag],'Color','w');

subplot(2,2,1)
plot(ts, opd, 'k', 'LineWidth',1.3);
xlabel(txlab)
ylabel(['OPD (', llab, ')'])
title(['OPD', tag])
grid on

subplot(2,2,2)
plot(ts, opdDotJ2,   'LineWidth',1.1); hold on
plot(ts, opdDotSRP,  'LineWidth',1.1);
plot(ts, opdDotMoon, 'LineWidth',1.1);
plot(ts, opdDotSun,  'LineWidth',1.1);
plot(ts, opdDotTot, 'k--', 'LineWidth',1.4);
xlabel(txlab)
ylabel(['OPD Rate (', rlab, ')'])
title(['OPD Rate Breakdown', tag])
legend('J2','SRP','Moon','Sun grav','Total','Location','best')
grid on

subplot(2,2,3)
plot(ts, dr_rtn(:,1), 'LineWidth',1.1); hold on
plot(ts, dr_rtn(:,2), 'LineWidth',1.1);
plot(ts, dr_rtn(:,3), 'LineWidth',1.1);
xlabel(txlab)
ylabel(['RTN Position (', llab, ')'])
title(['RTN Relative Position', tag])
legend('\delta r_R','\delta r_T','\delta r_N','Location','best')
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
    title(['Cumulative OPD Contribution', tag])
    legend('J2','SRP','Moon','Sun grav','Total','Location','best')
    grid on
else
    axis off
end

end

% =========================================================================
function s = local_extract_single_out(out, j)
s = struct();

s.roe = out.roe{j};
s.dr_rtn = out.dr_rtn{j};
s.opd = out.opd{j};

s.roe_dot.total = out.roe_dot.total{j};
s.roe_dot.J2    = out.roe_dot.J2{j};
s.roe_dot.SRP   = out.roe_dot.SRP{j};
s.roe_dot.Moon  = out.roe_dot.Moon{j};
s.roe_dot.Sun   = out.roe_dot.Sun{j};

s.opd_dot.total = out.opd_dot.total{j};
s.opd_dot.J2    = out.opd_dot.J2{j};
s.opd_dot.SRP   = out.opd_dot.SRP{j};
s.opd_dot.Moon  = out.opd_dot.Moon{j};
s.opd_dot.Sun   = out.opd_dot.Sun{j};

s.chief = out.chief;

s.deputy.rates.total = out.deputy.rates.total{j};
s.deputy.rates.J2    = out.deputy.rates.J2{j};
s.deputy.rates.SRP   = out.deputy.rates.SRP{j};
s.deputy.rates.Moon  = out.deputy.rates.Moon{j};
s.deputy.rates.Sun   = out.deputy.rates.Sun{j};

s.source_geom.total = out.source_geom.total{j};
s.source_geom.J2    = out.source_geom.J2{j};
s.source_geom.SRP   = out.source_geom.SRP{j};
s.source_geom.Moon  = out.source_geom.Moon{j};
s.source_geom.Sun   = out.source_geom.Sun{j};
end