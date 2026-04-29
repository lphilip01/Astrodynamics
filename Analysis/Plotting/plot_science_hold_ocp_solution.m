function plot_science_hold_ocp_solution(sol, opts)
% plot_science_hold_ocp_solution
%
% Visualize science-hold OCP solution.
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
for j = 1:Nc
    Uj = sol.U{j}; % (Nt-1) x 3
    uNorm(:,j) = sqrt(sum(Uj.^2,2));
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

% ---------------- Figure 2: Controls ----------------
figure('Name','Science Hold OCP - Controls','Color','w');

for j = 1:Nc
    subplot(Nc,2,2*j-1)
    plot(tt(1:end-1), sol.U{j}(:,1), 'LineWidth',1.1); hold on
    plot(tt(1:end-1), sol.U{j}(:,2), 'LineWidth',1.1);
    plot(tt(1:end-1), sol.U{j}(:,3), 'LineWidth',1.1);
    xlabel(tlabel)
    ylabel('u')
    title(sprintf('Collector %d RTN control', j))
    legend('u_R','u_T','u_N','Location','best')
    grid on

    subplot(Nc,2,2*j)
    plot(tt(1:end-1), uNorm(:,j), 'k', 'LineWidth',1.3)
    xlabel(tlabel)
    ylabel('||u||')
    title(sprintf('Collector %d control norm', j))
    grid on
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
fprintf('=============================================\n');

end