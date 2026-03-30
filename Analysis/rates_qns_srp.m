function dxdt = rates_qns_srp(t, x, params)
% rates_qns_srp
%
% Quasi-nonsingular SRP Gauss-rate model for near-circular Earth orbits.
%
% State:
%   x = [a; ex; ey; inc; RAAN; u]
%
% where
%   a    = semimajor axis [km]
%   ex   = e*cos(omega)
%   ey   = e*sin(omega)
%   inc  = inclination [rad]
%   RAAN = right ascension of ascending node [rad]
%   u    = omega + M  (mean argument of latitude) [rad]
%
% This uses the near-circular SRP rates:
%
%   a_dot     = -(2*pSR/n) * u_s
%   u_dot     =  n + (2*pSR/(n*a))*u_r + (pSR/(n*a))*cot(i)*sin(u)*u_w
%   ex_dot    = -(pSR/(n*a)) * ( sin(u)*u_r + 2*cos(u)*u_s )
%   ey_dot    =  (pSR/(n*a)) * ( cos(u)*u_r - 2*sin(u)*u_s )
%   i_dot     = -(pSR/(n*a)) * cos(u) * u_w
%   RAAN_dot  = -(pSR/(n*a*sin(i))) * sin(u) * u_w
%
% Inputs in params:
%   params.mu        gravitational parameter [km^3/s^2]
%   params.RE        Earth radius [km]
%   params.CR        reflectivity coefficient
%   params.As        illuminated area [m^2]
%   params.m         spacecraft mass [kg]
%   params.S         solar constant [W/m^2]
%   params.c         speed of light [m/s]
%   params.jd0       initial Julian date
%   params.ephemModel optional ephemeris model, e.g. '421'
%
% Requires:
%   los_qns.m  (shadow test)
%   Aerospace Toolbox planetEphemeris
%
% Notes:
%   1) This is a near-circular formulation.
%   2) u is treated as the fast angular variable.
%   3) planetEphemeris returns km for planetary positions.
%

% Unpack state
a    = x(1);   % km
ex   = x(2);
ey   = x(3);
inc  = x(4);   % rad
RAAN = x(5);   % rad
u    = x(6);   % rad

% Unpack parameters
mu = params.mu;
RE = params.RE;
CR = params.CR;
As = params.As;
m  = params.m;
S  = params.S;
c  = params.c;

if isfield(params,'ephemModel')
    eph = params.ephemModel;
else
    eph = '421';
end

% Current Julian date
jd = params.jd0 + t/86400;

% Sun position wrt Earth in J2000/ICRF [km]
r_sun = planetEphemeris(jd, 'Earth', 'Sun', eph);
r_sun = r_sun(:);

% Mean motion [rad/s]
n = sqrt(mu/a^3);

% Near-circular orbit geometry:
% For e ~ 0, radius and angular momentum
r = a;
h = sqrt(mu*a);

% RTN frame from (RAAN, inc, u)
cO = cos(RAAN); sO = sin(RAAN);
ci = cos(inc);  si = sin(inc);
cu = cos(u);    su = sin(u);

% R-hat (radial)
Rhat = [ cO*cu - sO*su*ci;
         sO*cu + cO*su*ci;
         su*si ];

% T-hat (along-track)
That = [ -cO*su - sO*cu*ci;
         -sO*su + cO*cu*ci;
          cu*si ];

% N-hat (orbit normal)
Nhat = [ sO*si;
        -cO*si;
         ci ];

% Satellite ECI position [km]
r_sat = r * Rhat;

% Eclipse / line-of-sight switch
nu_shadow = los_qns(r_sat, r_sun, RE);

% SRP acceleration magnitude [km/s^2]
% S/c is pressure [N/m^2], times CR*As/m gives [m/s^2], then /1000 to km/s^2
pSR = nu_shadow * (S/c) * CR * As / m / 1000;

% Sun direction unit vector in ECI
s_hat = r_sun / norm(r_sun);

% Components in RTN
u_r = dot(s_hat, Rhat);
u_s = dot(s_hat, That);
u_w = dot(s_hat, Nhat);

% Protect singularities near equatorial orbit
sin_i = sin(inc);
if abs(sin_i) < 1e-10
    sin_i = sign(sin_i + eps) * 1e-10;
end

tan_i = tan(inc);
if abs(tan_i) < 1e-10
    tan_i = sign(tan_i + eps) * 1e-10;
end

% Quasi-nonsingular SRP rates (near-circular)
a_dot    = -(2*pSR/n) * u_s;

u_dot    =  n ...
          + (2*pSR/(n*a))*u_r ...
          + (pSR/(n*a))*(cos(inc)/sin_i)*sin(u)*u_w;

ex_dot   = -(pSR/(n*a)) * ( sin(u)*u_r + 2*cos(u)*u_s );

ey_dot   =  (pSR/(n*a)) * ( cos(u)*u_r - 2*sin(u)*u_s );

inc_dot  = -(pSR/(n*a)) * cos(u) * u_w;

RAAN_dot = -(pSR/(n*a*sin_i)) * sin(u) * u_w;

% Return derivative
dxdt = [a_dot; ex_dot; ey_dot; inc_dot; RAAN_dot; u_dot];
end