function dxdt = rates_qns_j2(~, x, params)
% rates_qns_j2
%
% Near-circular quasi-nonsingular J2 rates for Earth orbit.
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
%   params.mu   gravitational parameter [km^3/s^2]
%   params.RE   Earth radius [km]
%   params.J2   J2 coefficient
%
% Notes:
%   1) This is a near-circular formulation.
%   2) For e ~ 0, u = omega + M is used as the fast angle and
%      approximated in the J2 RTN acceleration terms.
%   3) This reproduces the osculating short-period behavior at first order
%      in a QNS formulation without singular e,omega variables.
%

% Unpack state
a    = x(1);   % km
ex   = x(2);   %#ok<NASGU> % included for consistency; not needed explicitly here
ey   = x(3);   %#ok<NASGU>
inc  = x(4);   % rad
RAAN = x(5);   %#ok<NASGU> % J2 does not explicitly depend on RAAN here
u    = x(6);   % rad

% Unpack parameters
mu = params.mu;
RE = params.RE;
J2 = params.J2;

% Mean motion
n = sqrt(mu/a^3);

% Trig
si = sin(inc);
ci = cos(inc);
su = sin(u);
cu = cos(u);

% Circular-orbit J2 perturbing acceleration in RTN [km/s^2]
fac = -(3/2) * J2 * mu * RE^2 / a^4;

A_R = fac * (1 - 3*si^2*su^2);
A_T = fac * (si^2 * sin(2*u));
A_N = fac * (sin(2*inc) * su);

% Protect against equatorial singularity in RAAN rate
sin_i = sin(inc);
if abs(sin_i) < 1e-10
    sin_i = sign(sin_i + eps) * 1e-10;
end

% QNS rates
a_dot    = (2/n) * A_T;

u_dot    = n ...
         - (2/(n*a)) * A_R ...
         - (cos(inc)/(n*a*sin_i)) * sin(u) * A_N;

ex_dot   = (1/(n*a)) * ( sin(u)*A_R + 2*cos(u)*A_T );

ey_dot   = (1/(n*a)) * ( -cos(u)*A_R + 2*sin(u)*A_T );

inc_dot  = (1/(n*a)) * cos(u) * A_N;

RAAN_dot = (1/(n*a*sin_i)) * sin(u) * A_N;

% Return derivative
dxdt = [a_dot; ex_dot; ey_dot; inc_dot; RAAN_dot; u_dot];
end