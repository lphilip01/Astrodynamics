function [out, sol]=Example_GEO_Formation_Many(ra,dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,plot_figs,optimize)
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
e0    = 1e-4;
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
 gamma=0;
 delta_ex = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy = -rho*sin(gamma)*cos(beta);

  gamma=pi/4;
 delta_ex2 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey2 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy2 = -rho*sin(gamma)*cos(beta);

  gamma=2*pi/4;
 delta_ex3 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey3 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy3 = -rho*sin(gamma)*cos(beta);

  gamma=3*pi/4;
 delta_ex4 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey4 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy4 = -rho*sin(gamma)*cos(beta);

  gamma=4*pi/4;
 delta_ex5 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey5 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy5 = -rho*sin(gamma)*cos(beta);

  gamma=5*pi/4;
 delta_ex6 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey6 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy6 = -rho*sin(gamma)*cos(beta);

  gamma=6*pi/4;
 delta_ex7 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey7 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy7 = -rho*sin(gamma)*cos(beta);

  gamma=7*pi/4;
 delta_ex8 = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey8 = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy8 = -rho*sin(gamma)*cos(beta);

% Initial ROE from attached STARI relation
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

 roe04 = zeros(6,1);
 roe04(1) = 0.0;
 roe04(2) = 0.0;
 roe04(3) = delta_ex4/a0;
 roe04(4) = delta_ey4/a0;
 roe04(5) = 0.0;
 roe04(6) = delta_iy4/a0;

 roe06 = zeros(6,1);
 roe06(1) = 0.0;
 roe06(2) = 0.0;
 roe06(3) = delta_ex6/a0;
 roe06(4) = delta_ey6/a0;
 roe06(5) = 0.0;
 roe06(6) = delta_iy6/a0;

 roe05 = zeros(6,1);
 roe05(1) = 0.0;
 roe05(2) = 0.0;
 roe05(3) = delta_ex5/a0;
 roe05(4) = delta_ey5/a0;
 roe05(5) = 0.0;
 roe05(6) = delta_iy5/a0;

 roe07 = zeros(6,1);
 roe07(1) = 0.0;
 roe07(2) = 0.0;
 roe07(3) = delta_ex7/a0;
 roe07(4) = delta_ey7/a0;
 roe07(5) = 0.0;
 roe07(6) = delta_iy7/a0;

 roe08 = zeros(6,1);
 roe08(1) = 0.0;
 roe08(2) = 0.0;
 roe08(3) = delta_ex8/a0;
 roe08(4) = delta_ey8/a0;
 roe08(5) = 0.0;
 roe08(6) = delta_iy8/a0;

 fprintf('Initial GEO ROE d1:\n');
 fprintf('delta_a      = %.6e\n', roe0(1));
 fprintf('delta_lambda = %.6e\n', roe0(2));
 fprintf('delta_ex     = %.6e\n', roe0(3));
 fprintf('delta_ey     = %.6e\n', roe0(4));
 fprintf('delta_ix     = %.6e\n', roe0(5));
 fprintf('delta_iy     = %.6e\n', roe0(6));

 fprintf('Initial STARI ROE d2:\n');
 fprintf('delta_a      = %.6e\n', roe02(1));
 fprintf('delta_lambda = %.6e\n', roe02(2));
 fprintf('delta_ex     = %.6e\n', roe02(3));
 fprintf('delta_ey     = %.6e\n', roe02(4));
 fprintf('delta_ix     = %.6e\n', roe02(5));
 fprintf('delta_iy     = %.6e\n', roe02(6));

  fprintf('Initial STARI ROE d2:\n');
 fprintf('delta_a      = %.6e\n', roe02(1));
 fprintf('delta_lambda = %.6e\n', roe02(2));
 fprintf('delta_ex     = %.6e\n', roe02(3));
 fprintf('delta_ey     = %.6e\n', roe02(4));
 fprintf('delta_ix     = %.6e\n', roe02(5));
 fprintf('delta_iy     = %.6e\n', roe02(6));
 
%% ------------------------------------------------------------------------
% Build deputy initial QNS state from chief + ROE
%% ------------------------------------------------------------------------
xd0 = deputy_cell_from_chief_roe_qns(xc0, {roe0,roe02, roe03, roe04, roe05, roe06, roe07, roe08});

%% ------------------------------------------------------------------------
% Parameters: J2 only
%% ------------------------------------------------------------------------
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 4903;          % km^3/s^2 
paramsChief.muSun   = 132712;     % km^3/s^2 
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

paramsDeputy3 = paramsDeputy;
paramsDeputy4 = paramsDeputy;
paramsDeputy5 = paramsDeputy;
paramsDeputy6 = paramsDeputy;
paramsDeputy7 = paramsDeputy;
paramsDeputy8 = paramsDeputy;

%% ------------------------------------------------------------------------
% Time span: 5 orbits
%% ------------------------------------------------------------------------
T0 = 2*pi*sqrt(a0^3/mu);
nOrbits = 2;
tf = nOrbits * T0;

nout  = 3000;
tspan = linspace(0, tf, nout);

% Optional ephemeris precompute for compatibility with full pipeline
paramsChief.ephem  = precompute_ephemeris(tspan, paramsChief);
paramsDeputy.ephem = paramsChief.ephem;
paramsDeputy2.ephem = paramsChief.ephem;
paramsDeputy3.ephem = paramsChief.ephem;
paramsDeputy4.ephem = paramsChief.ephem;
paramsDeputy5.ephem = paramsChief.ephem;
paramsDeputy6.ephem = paramsChief.ephem;
paramsDeputy7.ephem = paramsChief.ephem;
paramsDeputy8.ephem = paramsChief.ephem;

opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

%% ------------------------------------------------------------------------
% Propagate chief and deputy
%% ------------------------------------------------------------------------
[t, xc] = ode45(@(t,x) rates_qns_total(t,x,paramsChief),  tspan, xc0, opts);
[~, xd] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy), tspan, xd0{1}, opts);
[~, xd2] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy2), tspan, xd0{2}, opts);
[~, xd3] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy3), tspan, xd0{3}, opts);
[~, xd4] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy4), tspan, xd0{4}, opts);
[~, xd5] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy5), tspan, xd0{5}, opts);
[~, xd6] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy6), tspan, xd0{6}, opts);
[~, xd7] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy7), tspan, xd0{7}, opts);
[~, xd8] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy8), tspan, xd0{8}, opts);

%% ------------------------------------------------------------------------
% Full OPD-rate pipeline
%% ------------------------------------------------------------------------
out = opd_pipeline_from_qns( ...
    t, xc, {xd,xd2,xd3,xd4,xd5,xd6,xd7,xd8}, ...
    paramsChief, {paramsDeputy,paramsDeputy2, paramsDeputy3,paramsDeputy4,paramsDeputy5,paramsDeputy6,paramsDeputy7,paramsDeputy8}, ...
    'phibeta', [phi0 beta]);
out.t=t;
xdCell={xd,xd2,xd3,xd4,xd5,xd6,xd7,xd8};
out.states.chief=xc;
out.states.deputies=xdCell;
out.star.phi0=phi0;
out.star.beta=beta;
out.star.ra=ra;
out.star.dec=dec;
% Suppose you already selected a science-hold start index k0
% and extracted chief and deputy states over the desired hold interval:
if optimize

[kStart, info] = find_science_hold_start(out, 5, 'bestLocal');

k0 = kStart-3;
Tint = Tint_s;   % 60 min
t0 = t(k0);
h_ocp = 30;
Npts  = max(round(Tint/h_ocp) + 1, 30);
tHold = linspace(0, Tint, Npts).';

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

target.type = 'phibeta';
target.phi0 = phi0;
target.beta = beta;

prob.Dmax_m = dmax;
prob.rhoMin_km = .8*rho;       % keep collectors far from combiner
prob.rhoMax_km = 1.2*rho;       % optional
prob.dPairMin_km = 0.5*rho;     % pairwise min spacing
prob.massDry_kg = .85*md;

sol = solve_science_hold_ocp_throttle_direction(tHold, chiefHold, depInit, paramsOCP, target, prob);
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


% RTN plot
figure();
hold on;
for i = 1:length(out.dr_rtn)
    plot3(out.dr_rtn{i}(:,2).*1000, out.dr_rtn{i}(:,3).*1000, out.dr_rtn{i}(:,1).*1000, ...
        'DisplayName', sprintf('Collector %d', i));
end
legend();
title("RTN for Collector Satellites");
ylabel("Normal Position");
xlabel("Tangential Position");
zlabel("Radial Position");
hold off

% OPD plot
figure();
hold on;
for i = 1:length(out.opd)
    plot(t./(60*60), out.opd{i}.*1000, 'DisplayName', sprintf('Collector %d', i));
end
legend();
title("OPD for Collector Satellites");
ylabel("OPD (m)");
xlabel("Time (hr)");
yline(0);
hold off
figure();plot(t./(60*60),out.opd{1}.*1000 + out.opd{2}.*1000+out.opd{3}.*1000,'DisplayName','Combined OPD');legend();title("OPD for Collector Satellites");ylabel("OPD (m)");xlabel("time");yline(0);

if optimize
figure();plot3(sol.RTN{1}(:,2).*1000,sol.RTN{1}(:,3).*1000,sol.RTN{1}(:,1).*1000,'DisplayName','Collector 1');hold on; plot3(sol.RTN{2}(:,2).*1000,sol.RTN{2}(:,3).*1000,sol.RTN{2}(:,1).*1000,'DisplayName','Collector 2');plot3(sol.RTN{3}(:,2).*1000,sol.RTN{3}(:,3).*1000,sol.RTN{3}(:,1).*1000,'DisplayName','Collector 3');legend();title("RTN for Collector Satellites");ylabel("Normal Position");xlabel("Tangential position");zlabel("Radial position");
end

end

metrics = formation_opd_metrics(out);

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
