% example_qns_free_time_ocp_setup
%
% Minimal setup template for solve_qns_free_time_ocp.m.
% Replace the placeholder data below with your chief history,
% ephemerides, deputy initial state, and dynamics parameters.

clear

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

deneb_ra=5.403;
deneb_dec=0.800;

[phi0, beta, sRTN0] = radec_to_phibeta(deneb_ra, deneb_dec, RAAN0, inc0);

rho_m=1000;
% Only one deputy described here out of 3, other deputies have gamma +
% 2*pi/3 and 4*pi/3
 rho=rho_m/1000;
 gamma=2*pi/3;
 delta_ex = rho*(cos(gamma)*sin(phi0)+sin(gamma)*sin(beta)*cos(phi0));
 delta_ey = -(rho/2)*(cos(gamma)*cos(phi0)-sin(gamma)*sin(beta)*sin(phi0));
 delta_iy = -rho*sin(gamma)*cos(beta);

% Initial ROE from attached STARI relation
 roe0 = zeros(6,1);
 roe0(1) = 0.0;
 roe0(2) = 0.0;
 roe0(3) = delta_ex/a0;
 roe0(4) = delta_ey/a0;
 roe0(5) = 0.0;
 roe0(6) = delta_iy/a0;

% fprintf('Initial GEO ROE d1:\n');
% fprintf('delta_a      = %.6e\n', roe0(1));
% fprintf('delta_lambda = %.6e\n', roe0(2));
% fprintf('delta_ex     = %.6e\n', roe0(3));
% fprintf('delta_ey     = %.6e\n', roe0(4));
% fprintf('delta_ix     = %.6e\n', roe0(5));
% fprintf('delta_iy     = %.6e\n', roe0(6));

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
xd0 = deputy_cell_from_chief_roe_qns(xc0, {roe0});

%% ------------------------------------------------------------------------
% Parameters: J2 only
%% ------------------------------------------------------------------------
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 4903;          % km^3/s^2 
paramsChief.muSun   = 132712;     % km^3/s^2 
paramsChief.CR      = 2; %2
paramsChief.As      = 2;           % m^2 3
paramsChief.m       = 200;           % kg 400
paramsChief.S       = 1367;          % W/m^2 1367
paramsChief.c       = 2.998e8;       % m/s
paramsChief.jd0     = juliandate(datetime(2026,1,1,0,0,0));
paramsChief.ephemModel = '421';
paramsChief.useShadow = true;

paramsDeputy = paramsChief;
paramsDeputy.As=2;
paramsDeputy.m=150;

%% ------------------------------------------------------------------------
% Time span: 2 orbits
%% ------------------------------------------------------------------------
T0 = 2*pi*sqrt(a0^3/mu);
nOrbits = 2;
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


% -------------------------------------------------------------------------
% Reference data on a physical time grid
% -------------------------------------------------------------------------
ref = struct();
ref.tGrid = t;          % [NtRef x 1] seconds
ref.chiefHist = xc;      % [NtRef x 6] [a ex ey i RAAN u]
ref.rSun = paramsChief.ephem.rSun;            % [NtRef x 3] km
ref.rMoon = paramsChief.ephem.rMoon;          % [NtRef x 3] km

% -------------------------------------------------------------------------
% Initial deputy state
% -------------------------------------------------------------------------
deputyInit = [xd0{1}(1); xd0{1}(2); xd0{1}(3); xd0{1}(4); xd0{1}(5); xd0{1}(6);paramsDeputy.m];

paramsOCP = paramsDeputy;
paramsOCP.T   = .2;   % N
paramsOCP.Isp = 1500;   % s



% -------------------------------------------------------------------------
% Problem definition
% -------------------------------------------------------------------------
prob = struct();

prob.N = 80;                    % number of shooting intervals
prob.TfMin = 1*3600;            % s
prob.TfMax = 20*3600;           % s
prob.TfGuess = 3*3600;          % s
prob.t0 = ref.tGrid(1);         % optional start time

prob.massDry_kg = 130.0;         % kg
prob.dMin_km = 0.25;            % km

% Desired terminal relative QNS geometry:
% [delta a; delta lambda; delta ex; delta ey; delta ix; delta iy]
prob.deltaAlphaTarget = [ ...
    0.0; ...
    0.0; ...
    2.0e-4; ...
   -1.0e-4; ...
    0.0; ...
    0.0];

% -------------------------------------------------------------------------
% Terminal weight matrix design examples
% -------------------------------------------------------------------------

% Option 1: isotropic weights
W_identity = eye(6);

% Option 2: reciprocal-variance scaling
sigma = [1e-4; 1e-4; 2e-5; 2e-5; 2e-5; 2e-5];
W_scaled = diag(1 ./ sigma.^2);

% Option 3: mission-priority weighting
W_priority = diag([1, 5, 20, 20, 10, 10]);

% Pick one:
prob.Wterm = W_scaled;

% -------------------------------------------------------------------------
% Objective weights
% -------------------------------------------------------------------------
prob.wf = 1.0;                  % final time weight
prob.wm = 1.0;                  % propellant use weight
prob.epsU = 1e-4;               % integral ||v||^2 regularization
prob.wterm = 1.0;               % terminal penalty multiplier

% -------------------------------------------------------------------------
% Outer-loop / solver options
% -------------------------------------------------------------------------
prob.maxOuterIter = 3;
prob.outerTol = 1e-3;
prob.interpMethod = 'pchip';
prob.unwrapChiefAngles = true;

prob.ipoptMaxIter = 600;
prob.ipoptTol = 1e-6;
prob.ipoptPrintLevel = 5;

% -------------------------------------------------------------------------
% Solve
% -------------------------------------------------------------------------
sol = solve_qns_free_time_ocp(ref, deputyInit, paramsOCP, prob);

fprintf('Solved: %d\n', sol.success);
fprintf('Tf = %.3f hr\n', sol.Tf_s / 3600);
fprintf('Mass used = %.6f kg\n', sol.massUsed_kg);
fprintf('Terminal error norm = %.6e\n', norm(sol.deltaAlphaError));

plot_qns_free_time_solution(sol);
