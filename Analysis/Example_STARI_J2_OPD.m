function Example_STARI_J2_OPD
% Example_STARI_J2_OPD
%
% End-to-end STARI validation example using the full OPD-rate pipeline:
%
%   chief/deputy QNS propagation
%   -> ROE
%   -> RTN baseline
%   -> OPD
%   -> OPD-rate breakdown by perturbation
%
% STARI case:
%   circular SS orbit
%   a = 6878 km
%   i = 97.402 deg
%   delta_lambda = 100 m
%   phi0 = 90 deg
%   beta = 45 deg
%
% This example uses J2 only, but runs through the full source-separated
% OPD-rate pipeline.

close all; clc;

%% ------------------------------------------------------------------------
% Constants
%% ------------------------------------------------------------------------
mu = 398600;          % km^3/s^2
RE = 6378;            % km
J2 = 1082.63e-6;
deg = pi/180;

%% ------------------------------------------------------------------------
% Chief orbit (STARI)
%% ------------------------------------------------------------------------
a0    = 6878;               % km
e0    = 0.0;
inc0  = 97.402 * deg;
RAAN0 = 0.0;
w0    = 0.0;
M0    = 0.0;
u0    = w0 + M0;

ex0 = e0*cos(w0);
ey0 = e0*sin(w0);

xc0 = [a0; ex0; ey0; inc0; RAAN0; u0];

%% ------------------------------------------------------------------------
% Target geometry
%% ------------------------------------------------------------------------
phi0 = 90 * deg;
beta = 45 * deg;

% STARI: delta_lambda = 100 m
delta_lambda_m  = 100;            % m
delta_lambda_km = delta_lambda_m / 1000;
delta_lambda    = delta_lambda_km / a0;   % dimensionless ROE form

% Initial ROE from attached STARI relation
roe0 = zeros(6,1);
roe0(1) = 0.0;
roe0(2) = delta_lambda;
roe0(3) = 0.0;
roe0(4) = 0.0;
roe0(5) = delta_lambda*cos(phi0)/tan(beta);
roe0(6) = delta_lambda*sin(phi0)/tan(beta);

fprintf('Initial STARI ROE:\n');
fprintf('delta_a      = %.6e\n', roe0(1));
fprintf('delta_lambda = %.6e\n', roe0(2));
fprintf('delta_ex     = %.6e\n', roe0(3));
fprintf('delta_ey     = %.6e\n', roe0(4));
fprintf('delta_ix     = %.6e\n', roe0(5));
fprintf('delta_iy     = %.6e\n', roe0(6));

%% ------------------------------------------------------------------------
% Build deputy initial QNS state from chief + ROE
%% ------------------------------------------------------------------------
xd0 = deputy_from_chief_roe_qns(xc0, roe0);

%% ------------------------------------------------------------------------
% Parameters: J2 only
%% ------------------------------------------------------------------------
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 0.0;
paramsChief.muSun   = 0.0;
paramsChief.CR      = 0.0;
paramsChief.As      = 0.0;
paramsChief.m       = 1.0;
paramsChief.S       = 0.0;
paramsChief.c       = 2.998e8;
paramsChief.jd0     = juliandate(datetime(2024,1,1,0,0,0));
paramsChief.ephemModel = '421';
paramsChief.useShadow = false;

paramsDeputy = paramsChief;

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

opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

%% ------------------------------------------------------------------------
% Propagate chief and deputy
%% ------------------------------------------------------------------------
[t, xc] = ode45(@(t,x) rates_qns_total(t,x,paramsChief),  tspan, xc0, opts);
[~, xd] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy), tspan, xd0, opts);

%% ------------------------------------------------------------------------
% Full OPD-rate pipeline
%% ------------------------------------------------------------------------
out = opd_pipeline_from_qns( ...
    t, xc, xd, ...
    paramsChief, paramsDeputy, ...
    'phibeta', [phi0 beta]);

%% ------------------------------------------------------------------------
% Plot full OPD-rate pipeline
%% ------------------------------------------------------------------------
plot_opd_pipeline_results(t, out)

%% ------------------------------------------------------------------------
% Summary
%% ------------------------------------------------------------------------
fprintf('\nOPD statistics over %.1f orbits:\n', nOrbits);
fprintf('  max(OPD)       = %+8.4f m\n', max(out.opd*1000));
fprintf('  min(OPD)       = %+8.4f m\n', min(out.opd*1000));
fprintf('  peak-to-peak   = %.4f m\n', max(out.opd*1000)-min(out.opd*1000));
fprintf('  max(|OPDdot|)  = %.6e m/s\n', max(abs(out.opd_dot.total*1000)));

end

% =========================================================================
function xd0 = deputy_from_chief_roe_qns(xc0, roe0)
% Build deputy QNS state from chief QNS state and near-circular ROE

ac  = xc0(1);
exc = xc0(2);
eyc = xc0(3);
ic  = xc0(4);
RAc = xc0(5);
uc  = xc0(6);

da   = roe0(1);
dl   = roe0(2);
dex  = roe0(3);
dey  = roe0(4);
dix  = roe0(5);
diy  = roe0(6);

ad  = ac * (1 + da);
exd = exc + dex;
eyd = eyc + dey;
id  = ic + dix;

sin_ic = sin(ic);
if abs(sin_ic) < 1e-12
    sin_ic = sign(sin_ic + eps) * 1e-12;
end

dRA   = diy / sin_ic;
RAANd = RAc + dRA;
ud    = uc + dl - dRA*cos(ic);

xd0 = [ad; exd; eyd; id; RAANd; ud];
end