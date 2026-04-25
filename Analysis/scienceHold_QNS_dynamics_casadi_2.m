function xdot = scienceHold_QNS_dynamics_casadi(x, vctrl, params, rSun, rMoon)
% scienceHold_QNS_dynamics_casadi
%
% CasADi-compatible QNS dynamics for a single spacecraft with:
%   - J2
%   - SRP
%   - lunar gravity
%   - solar gravity
%   - electric propulsion control in RTN
%
% State:
%   x = [a; ex; ey; inc; RAAN; u; m]
%
% Control:
%   vctrl = [vR; vT; vN], with norm(vctrl) <= 1
%
% Notes:
%   - "params" is expected to contain spacecraft-specific propulsion and
%     area properties when used inside a multi-spacecraft OCP.
%   - The function remains compatible with the previous single-parameter
%     struct interface.

mu     = local_get_param(params, 'mu', 0);
RE     = local_get_param(params, 'RE', 0);
J2     = local_get_param(params, 'J2', 0);
muMoon = local_get_param(params, 'muMoon', 0);
muSun  = local_get_param(params, 'muSun', 0);
CR     = local_get_param(params, 'CR', 0);
As     = local_get_param(params, 'As', 0);
S      = local_get_param(params, 'S', 0);
c      = local_get_param(params, 'c', 299792458);
T      = local_get_param(params, 'T', 0);
Isp    = local_get_param(params, 'Isp', inf);
g0     = local_get_param(params, 'g0', 9.80665);

a    = x(1);
ex   = x(2); %#ok<NASGU>
ey   = x(3); %#ok<NASGU>
inc  = x(4);
RAAN = x(5);
u    = x(6);
m    = x(7);

vR = vctrl(1);
vT = vctrl(2);
vN = vctrl(3);

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

% Moon gravity
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
if T == 0
    athrust = 0;
else
    athrust = (T / m) / 1000;   % km/s^2
end

AR_th = athrust * vR;
AT_th = athrust * vT;
AN_th = athrust * vN;

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

% Smooth non-negative throttle: always >= 0, smooth derivative everywhere
eps_th   = 1e-10;
norm_sq  = vR^2 + vT^2 + vN^2;
throttle = sqrt(norm_sq + eps_th^2) - eps_th;

if (T == 0) || ~isfinite(Isp) || (Isp <= 0)
    m_dot = 0;
else
    m_dot = -(T * throttle) / (Isp * g0);
end

xdot = [a_dot; ex_dot; ey_dot; inc_dot; RAAN_dot; u_dot; m_dot];
end

function value = local_get_param(params, name, defaultValue)
if isfield(params, name) && ~isempty(params.(name))
    value = params.(name);
else
    value = defaultValue;
end
end
