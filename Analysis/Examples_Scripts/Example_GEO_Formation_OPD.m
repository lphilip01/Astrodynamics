function [out, sol]=Example_GEO_Formation_OPD(ra,dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,settle_time,plot_figs,optimize)
% Example_GEO_Linear_OPD
%
% End-to-end STARI validation example using the full OPD-rate pipeline:
%
%   chief/deputy QNS propagation
%   -> ROE
%   -> RTN baseline
%   -> OPD
%   -> OPD-rate breakdown by perturbation
%
%


close all; clc;

%% ------------------------------------------------------------------------
% Constants
%% ------------------------------------------------------------------------
mu = 398600;          % km^3/s^2
RE = 6378;            % km
J2 = 1082.63e-6;
deg = pi/180;

%% ------------------------------------------------------------------------
% Chief orbit (GEO, frozen)
%% ------------------------------------------------------------------------
a0    = 42164;               % km
e0    = 0;
inc0  = inc * deg;
RAAN0 = 0.0* deg;
w0    = 0.0* deg;
M0    = 0.0;
u0    = w0 + M0;

ex0 = e0*cos(w0);
ey0 = e0*sin(w0);

xc0 = [a0; ex0; ey0; inc0; RAAN0; u0];

%% ------------------------------------------------------------------------
% Target geometry
%% ------------------------------------------------------------------------



[phi0, beta, sRTN0] = radec_to_phibeta(ra, dec, RAAN0, inc0);


% Only one deputy described here out of 3, other deputies have gamma +
% 2*pi/3 and 4*pi/3
 rho=rho_m/1000;
 gamma=2*pi/3 +pi/4;
 delta_ex = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy = -rho*sin(gamma)*cos(beta);

  gamma=4*pi/3+pi/4;
 delta_ex2 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey2 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy2 = -rho*sin(gamma)*cos(beta);

  gamma=6*pi/3+pi/4;
 delta_ex3 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey3 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy3 = -rho*sin(gamma)*cos(beta);

 roe0 = zeros(6,1);
 roe0(1) = 0.0;
 roe0(2) = 0.0;
 roe0(3) = delta_ex/a0;
 roe0(4) = delta_ey/a0;
 roe0(5) = 0.0;
 roe0(6) = delta_iy/a0;

 roe02 = zeros(6,1);
 roe02(1) = 0.0;
 roe02(2) = 0.0;
 roe02(3) = delta_ex2/a0;
 roe02(4) = delta_ey2/a0;
 roe02(5) = 0.0;
 roe02(6) = delta_iy2/a0;

 roe03 = zeros(6,1);
 roe03(1) = 0.0;
 roe03(2) = 0.0;
 roe03(3) = delta_ex3/a0;
 roe03(4) = delta_ey3/a0;
 roe03(5) = 0.0;
 roe03(6) = delta_iy3/a0;

%% ------------------------------------------------------------------------
% Build deputy initial QNS state from chief + ROE
%% ------------------------------------------------------------------------
xd0 = deputy_cell_from_chief_roe_qns(xc0, {roe0,roe02, roe03});

%% ------------------------------------------------------------------------
% Parameters: J2 only
%% ------------------------------------------------------------------------
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 4903;          % km^3/s^2 
paramsChief.muSun   = 1.3271244*10^11;     % km^3/s^2 
paramsChief.CR      = 1.5; %2
paramsChief.As      = Asc;           % m^2 3
paramsChief.m       = mc;           % kg 400
paramsChief.S       = 1367;          % W/m^2 1367
paramsChief.c       = 2.998e8;       % m/s
paramsChief.jd0     = juliandate(datetime(2026,1,1,0,0,0));
paramsChief.ephemModel = '421';
paramsChief.useShadow = true;

paramsDeputy = paramsChief;
paramsDeputy.As=Asd;
paramsDeputy.m=md;

paramsDeputy2 = paramsChief;
paramsDeputy2.As=Asd;
paramsDeputy2.m=md;

paramsDeputy3 = paramsChief;
paramsDeputy3.As=Asd;
paramsDeputy3.m=md;

%% ------------------------------------------------------------------------
% Time span: 5 orbits
%% ------------------------------------------------------------------------
T0 = 2*pi*sqrt(a0^3/mu);
nOrbits = 3;
tf = nOrbits * T0;

nout  = 3000;
tspan = linspace(0, tf, nout);

% Optional ephemeris precompute for compatibility with full pipeline
paramsChief.ephem  = precompute_ephemeris(tspan, paramsChief);
paramsDeputy.ephem = paramsChief.ephem;
paramsDeputy2.ephem = paramsChief.ephem;
paramsDeputy3.ephem = paramsChief.ephem;

opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

%% ------------------------------------------------------------------------
% Propagate chief and deputy
%% ------------------------------------------------------------------------
[t, xc] = ode45(@(t,x) rates_qns_total(t,x,paramsChief),  tspan, xc0, opts);
[~, xd] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy), tspan, xd0{1}, opts);
[~, xd2] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy2), tspan, xd0{2}, opts);
[~, xd3] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy3), tspan, xd0{3}, opts);

%% ------------------------------------------------------------------------
% Full OPD-rate pipeline
%% ------------------------------------------------------------------------
out = opd_pipeline_from_qns( ...
    t, xc, {xd,xd2,xd3}, ...
    paramsChief, {paramsDeputy,paramsDeputy2, paramsDeputy3}, ...
    'phibeta', [phi0 beta]);
out.t=t;
xdCell={xd,xd2,xd3};
out.states.chief=xc;
out.states.deputies=xdCell;
out.star.phi0=phi0;
out.star.beta=beta;
out.star.ra=ra;
out.star.dec=dec;

% Complete star -> collector -> combiner path, relative to a common
% star -> combiner path. Units: km.
chiefGeometry = struct( ...
    'inc', xc(:,4), 'RAAN', xc(:,5), 'u', xc(:,6));

out.OPDfull_km = zeros(numel(t), numel(xdCell));
for j = 1:numel(xdCell)
    roe = roe_from_qns_chief_deputy(xc, xdCell{j});
    dr  = rtn_from_roe(roe, xc(:,1), xc(:,6));
    external_km = opd_from_rtn(dr, 'radec', [ra dec], chiefGeometry);
    out.OPDfull_km(:,j) = sqrt(sum(dr.^2, 2)) - external_km;
end
out.OPDfullSpread_m = 1000 * ...
    (max(out.OPDfull_km, [], 2) - min(out.OPDfull_km, [], 2));

out.perturbations = differential_perturbation_report( ...
    out, paramsChief, {paramsDeputy,paramsDeputy2,paramsDeputy3}, plot_figs);


if optimize

settle_s = settle_time; 

[k0, info] = find_science_hold_start( ...
    out, dmax, T0, settle_s, Tint_s);

t0 = t(k0);
nSettle = ceil(info.settleDuration_s / 30);
h_ocp = info.settleDuration_s / nSettle;
nScience = ceil(Tint_s / h_ocp);

tHold = (0:nSettle+nScience).' * h_ocp;
if t0 + tHold(end) > t(end)
    error('Extend the propagation to cover settling and science.');
end

prob.scienceStartNode = nSettle + 1;

% Interpolate chief onto hold grid
chiefHold = interp1(t - t0, xc, tHold, 'linear');

% Build deputy initial state cell (append mass)
m0 = md; % kg
depInit = cell(1,3);
for j = 1:3
    x0j = interp1(t - t0, xdCell{j}, 0, 'linear').';
    depInit{j} = [x0j; m0];
end

paramsOCP = paramsDeputy;
paramsOCP.T   = T;   % N
paramsOCP.Isp = 1500;   % s

% Ephemerides over hold grid
paramsOCP.ephem.t = tHold;
paramsOCP.ephem.rSun  = interp1(t - t0, paramsChief.ephem.rSun,  tHold, 'linear');
paramsOCP.ephem.rMoon = interp1(t - t0, paramsChief.ephem.rMoon, tHold, 'linear');

target.type = 'radec';
target.ra = ra;
target.dec = dec;

prob.Dmax_m = dmax;
prob.rhoMin_km = .8*rho;       % keep collectors far from combiner
prob.rhoMax_km = 1.2*rho;       % optional
prob.dPairMin_km = 0.5*rho;     % pairwise min spacing
prob.massDry_kg = .85*md;

sol = solve_science_hold_ocp_throttle_direction(tHold, chiefHold, depInit, paramsOCP, target, prob);

sol.scienceStartNode = prob.scienceStartNode;
sol.info   = info;        % handoff/alignment times from find_science_hold_start
sol.target = target;

else
sol=[];
end
%% ------------------------------------------------------------------------
% Plot full OPD-rate pipeline
%% ------------------------------------------------------------------------


if plot_figs

plot_opd_pipeline_results(t, out)

if optimize
plot_science_hold_ocp_solution(sol)
end


figure('Units','inches','Position',[1 1 5 4.2]);

% Color palette (colorblind-friendly, print-safe)
colors = [0.0000, 0.4470, 0.7410;   % blue
          0.8500, 0.3250, 0.0980;   % red-orange
          0.4660, 0.6740, 0.1880];  % green

lineStyles = {'-', '-', '-'};
nCollectors = 3;

ax = axes();
hold(ax, 'on');

for i = 1:nCollectors
    T = out.dr_rtn{i}(:,2);   % Tangential [m]
    N = out.dr_rtn{i}(:,3);   % Normal     [m]
    R = out.dr_rtn{i}(:,1);   % Radial     [m]

    ps=plot3(ax, T(1), R(1), N(1),'o', 'MarkerSize', 8,...
        'Color',      colors(i,:), ...
        'MarkerFaceColor', colors(i,:));
    set(ps, 'HandleVisibility', 'off');

    plot3(ax, T, R, N, ...
        'LineStyle',  lineStyles{i}, ...
        'LineWidth',  2, ...
        'Color',      colors(i,:), ...
        'DisplayName', sprintf('Collector %d', i));
end

hold(ax, 'off');

%% Axis labels (with units)
xlabel(ax, 'Tangential Position [km]', 'Interpreter', 'latex');
ylabel(ax, 'Radial Position [km]',     'Interpreter', 'latex');
zlabel(ax, 'Normal Position [km]',     'Interpreter', 'latex');

%% Legend
lgd = legend(ax, 'Location', 'best', 'Interpreter', 'latex','Box','on');
lgd.FontSize = 10;

%% Font & tick formatting
set(ax, ...
    'FontName',         'Times New Roman', ...
    'FontSize',         12, ...
    'FontWeight','bold',...
    'TickLabelInterpreter', 'latex', ...
    'LineWidth',        0.8, ...
    'Box',              'on', ...
    'XMinorTick',       'on', ...
    'YMinorTick',       'on', ...
    'ZMinorTick',       'on', ...
    'TickDir',          'in', ...
    'GridLineStyle',    '--', ...
    'GridAlpha',        0.3);

grid(ax, 'on');

%% View angle — adjust to best reveal your formation geometry
view(ax, 35, 25);

%% Export at 600 DPI (IEEE/AIAA requirement)
exportgraphics(gcf, 'rtn_trajectories.pdf', ...
    'ContentType', 'vector', 'BackgroundColor', 'none');

exportgraphics(gcf, 'rtn_trajectories.png', ...
    'Resolution', 600, 'BackgroundColor', 'white');
figure();plot(t./(60*60),out.opd{1}.*1000,'DisplayName','Collector 1');hold on; plot(t./(60*60),out.opd{2}.*1000,'DisplayName', 'Collector 2');plot(t./(60*60),out.opd{3}.*1000,'DisplayName', 'Collector 3');legend();title("OPD for Collector Satellites");ylabel("OPD (m)");xlabel("time");yline(0);

figure();plot(t./(60*60),out.opd{1}.*1000 + out.opd{2}.*1000+out.opd{3}.*1000,'DisplayName','Combined OPD');legend();title("OPD for Collector Satellites");ylabel("OPD (m)");xlabel("time");yline(0);

if optimize
figure();plot3(sol.RTN{1}(:,2).*1000,sol.RTN{1}(:,3).*1000,sol.RTN{1}(:,1).*1000,'DisplayName','Collector 1');hold on; plot3(sol.RTN{2}(:,2).*1000,sol.RTN{2}(:,3).*1000,sol.RTN{2}(:,1).*1000,'DisplayName','Collector 2');plot3(sol.RTN{3}(:,2).*1000,sol.RTN{3}(:,3).*1000,sol.RTN{3}(:,1).*1000,'DisplayName','Collector 3');legend();title("RTN for Collector Satellites");ylabel("Normal Position");xlabel("Tangential position");zlabel("Radial position");
end

end

metrics = formation_opd_metrics(num2cell(out.OPDfull_km,1));

fprintf('\nMax formation OPD range: %.3f m\n', metrics.summary.maxRangeOPD_m);
fprintf('Max formation RMS relative OPD: %.3f m\n', metrics.summary.maxRMSRelativeOPD_m);
fprintf('Worst collector index: %d\n', metrics.summary.worstCollectorIndex);

%% ------------------------------------------------------------------------
% Summary
%% ------------------------------------------------------------------------
fprintf('\nOPD statistics over %.1f orbits:\n', nOrbits);
fprintf('  max(OPD)       = %+8.4f m\n', (max(out.opd{1})*1000));
fprintf('  min(OPD)       = %+8.4f m\n', min((out.opd{1})*1000));
fprintf('  peak-to-peak   = %.4f m\n', max((out.opd{1})*1000)-min((out.opd{1})*1000));
fprintf('  max(|OPDdot|)  = %.6e m/s\n', max(abs((out.opd_dot.total{1})*1000)));

end
