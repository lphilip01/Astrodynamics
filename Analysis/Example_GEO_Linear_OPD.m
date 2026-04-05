function out=Example_GEO_Linear_OPD
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
inc0  = 7.4 * deg;
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

tau_ceti_ra=0.4600;
tau_ceti_dec = -0.2781;

altair_ra=5.1832;
altair_dec=0.1548;

deneb_ra=5.403;
deneb_dec=0.800;


[phi0, beta, sRTN0] = radec_to_phibeta(tau_ceti_ra, tau_ceti_dec, RAAN0, inc0);


% STARI: delta_lambda = 100 m
delta_lambda_m  = 1000;            % m
delta_lambda_km = delta_lambda_m / 1000;
delta_lambda    = delta_lambda_km / a0;   % dimensionless ROE form

% Only one deputy described here out of 3, other deputies have gamma +
% 2*pi/3 and 4*pi/3
% rho=5000/1000;
% gamma=2*pi/3;
% delta_ex = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
% delta_ey = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
% delta_iy = -rho*sin(gamma)*cos(beta);

% Initial ROE from attached STARI relation
% roe0 = zeros(6,1);
% roe0(1) = 0.0;
% roe0(2) = 0.0;
% roe0(3) = delta_ex/a0;
% roe0(4) = delta_ey/a0;
% roe0(5) = 0.0;
% roe0(6) = delta_iy/a0;

 roe0 = zeros(6,1);
 roe0(1) = 0.0;
 roe0(2) = -delta_lambda;
 roe0(3) = 0.0;
 roe0(4) = 0.0;
 roe0(5) = -delta_lambda*cos(phi0)/tan(beta);
 roe0(6) = -delta_lambda*sin(phi0)/tan(beta);

 roe02 = zeros(6,1);
 roe02(1) = 0.0;
 roe02(2) = delta_lambda;
 roe02(3) = 0.0;
 roe02(4) = 0.0;
 roe02(5) = delta_lambda*cos(phi0)/tan(beta);
 roe02(6) = delta_lambda*sin(phi0)/tan(beta);

fprintf('Initial GEO ROE d1:\n');
fprintf('delta_a      = %.6e\n', roe0(1));
fprintf('delta_lambda = %.6e\n', roe0(2));
fprintf('delta_ex     = %.6e\n', roe0(3));
fprintf('delta_ey     = %.6e\n', roe0(4));
fprintf('delta_ix     = %.6e\n', roe0(5));
fprintf('delta_iy     = %.6e\n', roe0(6));

% fprintf('Initial STARI ROE d2:\n');
% fprintf('delta_a      = %.6e\n', roe02(1));
% fprintf('delta_lambda = %.6e\n', roe02(2));
% fprintf('delta_ex     = %.6e\n', roe02(3));
% fprintf('delta_ey     = %.6e\n', roe02(4));
% fprintf('delta_ix     = %.6e\n', roe02(5));
% fprintf('delta_iy     = %.6e\n', roe02(6));

%% ------------------------------------------------------------------------
% Build deputy initial QNS state from chief + ROE
%% ------------------------------------------------------------------------
xd0 = deputy_cell_from_chief_roe_qns(xc0, {roe0,roe02});

%% ------------------------------------------------------------------------
% Parameters: J2 only
%% ------------------------------------------------------------------------
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 4903;          % km^3/s^2 
paramsChief.muSun   = 132712;     % km^3/s^2 
paramsChief.CR      = 2; %2
paramsChief.As      = 3;           % m^2 3
paramsChief.m       = 400;           % kg 400
paramsChief.S       = 1367;          % W/m^2 1367
paramsChief.c       = 2.998e8;       % m/s
paramsChief.jd0     = juliandate(datetime(2026,1,1,0,0,0));
paramsChief.ephemModel = '421';
paramsChief.useShadow = true;

paramsDeputy = paramsChief;
paramsDeputy.As=3;
paramsDeputy.m=400;

paramsDeputy2 = paramsChief;
paramsDeputy2.As=3;
paramsDeputy2.m=400;

%% ------------------------------------------------------------------------
% Time span: 5 orbits
%% ------------------------------------------------------------------------
T0 = 2*pi*sqrt(a0^3/mu);
nOrbits = 5;
tf = nOrbits * T0;

nout  = 3000;
tspan = linspace(0, tf, nout);

% Optional ephemeris precompute for compatibility with full pipeline
paramsChief.ephem  = precompute_ephemeris(tspan, paramsChief);
paramsDeputy.ephem = paramsChief.ephem;
paramsDeputy2.ephem = paramsChief.ephem;

opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

%% ------------------------------------------------------------------------
% Propagate chief and deputy
%% ------------------------------------------------------------------------
[t, xc] = ode45(@(t,x) rates_qns_total(t,x,paramsChief),  tspan, xc0, opts);
[~, xd] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy), tspan, xd0{1}, opts);
[~, xd2] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy2), tspan, xd0{2}, opts);

%% ------------------------------------------------------------------------
% Full OPD-rate pipeline
%% ------------------------------------------------------------------------
out = opd_pipeline_from_qns( ...
    t, xc, {xd,xd2}, ...
    paramsChief, {paramsDeputy,paramsDeputy2}, ...
    'phibeta', [phi0 beta]);

%% ------------------------------------------------------------------------
% Plot full OPD-rate pipeline
%% ------------------------------------------------------------------------
plot_opd_pipeline_results(t, out)

figure();plot3(out.dr_rtn{1}(:,2).*1000,out.dr_rtn{1}(:,3).*1000,out.dr_rtn{1}(:,1).*1000,'DisplayName','Collector 1');hold on; plot3(out.dr_rtn{2}(:,2).*1000,out.dr_rtn{2}(:,3).*1000,out.dr_rtn{2}(:,1).*1000,'DisplayName','Collector 2');legend();title("RTN for Collector Satellites");ylabel("Normal Position");xlabel("Tangential position");zlabel("Radial position");
figure();plot(out.opd{1}.*1000,'DisplayName','Collector 1');hold on; plot(out.opd{2}.*1000,'DisplayName', 'Collector 2');legend();title("OPD for Collector Satellites");ylabel("OPD (m)");xlabel("time step");



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
