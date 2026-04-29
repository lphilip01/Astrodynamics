function xdot = scienceHold_QNS_dynamics_casadi_throttle_direction(x, throttle, thrustDirRTN, params, rSun, rMoon)
% scienceHold_QNS_dynamics_casadi_throttle_direction
%
% CasADi-compatible QNS dynamics for one collector satellite with:
%   - J2
%   - SRP
%   - lunar gravity
%   - solar gravity
%   - electric propulsion split into scalar throttle and RTN direction
%
% State:
%   x = [a; ex; ey; inc; RAAN; u; m]
%
% Control:
%   throttle     = scalar alpha in [0, 1]
%   thrustDirRTN = [dR; dT; dN], intended to be unit norm

mu     = params.mu;
RE     = params.RE;
J2     = params.J2;
muMoon = params.muMoon;
muSun  = params.muSun;
CR     = params.CR;
As     = params.As;
S      = params.S;
c      = params.c;
T      = params.T;
Isp    = params.Isp;
g0     = 9.80665;

a    = x(1);
ex   = x(2);
ey   = x(3);
inc  = x(4);
RAAN = x(5);
u    = x(6);
m    = x(7);

alpha = throttle(1);
dR = thrustDirRTN(1);
dT = thrustDirRTN(2);
dN = thrustDirRTN(3);

n = sqrt(mu / a^3);
r = a;

cO = cos(RAAN); sO = sin(RAAN);
ci = cos(inc);  si = sin(inc);
cu = cos(u);    su = sin(u);

Rhat = [ cO*cu - sO*su*ci;
         sO*cu + cO*su*ci;
         su*si ];

That = [ -cO*su - sO*cu*ci;
         -sO*su + cO*cu*ci;
          cu*si ];

Nhat = [ -sO*si;
          cO*si;
         -ci ];

r_sat = r * Rhat;

% J2
facJ2 = -(3/2) * J2 * mu * RE^2 / a^4;
AR_J2 = facJ2 * (1 - 3*si^2*su^2);
AT_J2 = facJ2 * (si^2 * sin(2*u));
AN_J2 = facJ2 * (sin(2*inc) * su);

% SRP
if (As == 0) || (CR == 0) || (S == 0)
    AR_SRP = 0; AT_SRP = 0; AN_SRP = 0;
else
    s_hat = rSun / sqrt(sum(rSun.^2));
    pSR = (S/c) * CR * As / m / 1000;   % km/s^2
    AR_SRP = -pSR * dot(s_hat, Rhat);
    AT_SRP = -pSR * dot(s_hat, That);
    AN_SRP = -pSR * dot(s_hat, Nhat);
end

% Moon
if muMoon == 0
    AR_Moon = 0; AT_Moon = 0; AN_Moon = 0;
else
    rhoMoon = rMoon - r_sat;
    aMoonECI = muMoon * ( rhoMoon / (sqrt(sum(rhoMoon.^2))^3) ...
                        - rMoon  / (sqrt(sum(rMoon.^2))^3) );
    AR_Moon = dot(aMoonECI, Rhat);
    AT_Moon = dot(aMoonECI, That);
    AN_Moon = dot(aMoonECI, Nhat);
end

% Sun gravity
if muSun == 0
    AR_Sun = 0; AT_Sun = 0; AN_Sun = 0;
else
    rhoSun = rSun - r_sat;
    aSunECI = muSun * ( rhoSun / (sqrt(sum(rhoSun.^2))^3) ...
                      - rSun  / (sqrt(sum(rSun.^2))^3) );
    AR_Sun = dot(aSunECI, Rhat);
    AT_Sun = dot(aSunECI, That);
    AN_Sun = dot(aSunECI, Nhat);
end

% Thrust
athrust = (T / m) / 1000;   % km/s^2
AR_th = athrust * alpha * dR;
AT_th = athrust * alpha * dT;
AN_th = athrust * alpha * dN;

AR = AR_J2 + AR_SRP + AR_Moon + AR_Sun + AR_th;
AT = AT_J2 + AT_SRP + AT_Moon + AT_Sun + AT_th;
AN = AN_J2 + AN_SRP + AN_Moon + AN_Sun + AN_th;

sin_i_safe = sin(inc) + 1e-8;

a_dot    = (2/n) * AT;
u_dot    = n - (2/(n*a))*AR - (cos(inc)/(n*a*sin_i_safe)) * sin(u) * AN;
ex_dot   = (1/(n*a)) * ( sin(u)*AR + 2*cos(u)*AT );
ey_dot   = (1/(n*a)) * ( -cos(u)*AR + 2*sin(u)*AT );
inc_dot  = (1/(n*a)) * cos(u) * AN;
RAAN_dot = (1/(n*a*sin_i_safe)) * sin(u) * AN;

% Mass flow depends on commanded throttle directly.
m_dot = -(T * alpha) / (Isp * g0);

xdot = [a_dot; ex_dot; ey_dot; inc_dot; RAAN_dot; u_dot; m_dot];
end
