%% Earth Parameters
mu_earth = 398600.4418;        % km^3/s^2    (GMAT default, EGM-96)
R_earth  = 6378.137;           % km          (WGS-84 equatorial radius)
J2       = 1.08262668e-3;      % dimensionless (EGM-96)

%% GEO Orbit Parameters
a_GEO    = 42164.17;           % km          (semimajor axis for geostationary)
n_GEO    = sqrt(mu_earth / a_GEO^3);  % rad/s  (~7.2921e-5 rad/s)
P_GEO    = 2*pi / n_GEO;       % s           (~86164 s = 23h 56m 4s)

%% Solar Parameters
mu_sun   = 1.32712440018e11;   % km^3/s^2    (JPL DE430)
a_sun    = 1.49597870700e8;    % km          (1 AU, IAU 2012)
P_srp    = 4.56e-6;            % N/m^2       (solar radiation pressure at 1 AU)

%% Lunar Parameters
mu_moon  = 4902.8000;          % km^3/s^2    (GMAT default)
a_moon   = 384400;             % km          (mean Earth-Moon distance)

%% Example Spacecraft Parameters (adjust for your design)
mass_sc  = 200;                % kg          (typical smallsat)
A_sc     = 2.0;                % m^2         (cross-sectional area)
C_R      = 1.5;                % dimensionless (radiation pressure coefficient)
B_s      = C_R * A_sc / mass_sc;  % m^2/kg   (~0.015 m^2/kg)
delta_Bs = 0.10 * B_s;         % m^2/kg      (10% mismatch between combiner/collector)

%% Laplace Plane Inclination at GEO (J2 + lunisolar equilibrium)
inc_Laplace = 7.4 * pi/180;    % rad         (from Ito 2024 Eq. 28)