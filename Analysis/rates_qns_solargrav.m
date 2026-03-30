function dxdt = rates_qns_solargrav(t, x, params)
% rates_qns_solargrav
%
% Near-circular quasi-nonsingular solar third-body rates for Earth orbit.
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
% Inputs in params:
%   params.mu         Earth gravitational parameter [km^3/s^2]
%   params.muSun      Sun gravitational parameter [km^3/s^2]
%   params.jd0        initial Julian date
%   params.ephemModel optional ephemeris model, e.g. '421'
%
% Notes:
%   1) Near-circular formulation.
%   2) Satellite position is built from circular geometry using (a,i,RAAN,u).
%   3) Sun position is obtained from planetEphemeris.
%

% Unpack state
a    = x(1);   % km
ex   = x(2);   %#ok<NASGU> % not explicitly used in circular geometry
ey   = x(3);   %#ok<NASGU>
inc  = x(4);   % rad
RAAN = x(5);   % rad
u    = x(6);   % rad

% Parameters
mu    = params.mu;
muSun = params.muSun;

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

% Mean motion
n = sqrt(mu/a^3);

% Near-circular orbit geometry
r = a;

cO = cos(RAAN); sO = sin(RAAN);
ci = cos(inc);  si = sin(inc);
cu = cos(u);    su = sin(u);

% RTN basis vectors in ECI
Rhat = [ cO*cu - sO*su*ci;
         sO*cu + cO*su*ci;
         su*si ];

That = [ -cO*su - sO*cu*ci;
         -sO*su + cO*cu*ci;
          cu*si ];

Nhat = [ sO*si;
        -cO*si;
         ci ];

% Spacecraft position [km]
r_sat = r * Rhat;

% Exact solar third-body differential acceleration [km/s^2]
rho = r_sun - r_sat;
a_3b = muSun * ( rho/norm(rho)^3 - r_sun/norm(r_sun)^3 );

% RTN components
A_R = dot(a_3b, Rhat);
A_T = dot(a_3b, That);
A_N = dot(a_3b, Nhat);

% Protect against equatorial singularity
sin_i = sin(inc);
if abs(sin_i) < 1e-10
    sin_i = sign(sin_i + eps) * 1e-10;
end

% QNS near-circular rates
a_dot    = (2/n) * A_T;

u_dot    = n ...
         - (2/(n*a)) * A_R ...
         - (cos(inc)/(n*a*sin_i)) * sin(u) * A_N;

ex_dot   = (1/(n*a)) * ( sin(u)*A_R + 2*cos(u)*A_T );

ey_dot   = (1/(n*a)) * ( -cos(u)*A_R + 2*sin(u)*A_T );

inc_dot  = (1/(n*a)) * cos(u) * A_N;

RAAN_dot = (1/(n*a*sin_i)) * sin(u) * A_N;

dxdt = [a_dot; ex_dot; ey_dot; inc_dot; RAAN_dot; u_dot];
end