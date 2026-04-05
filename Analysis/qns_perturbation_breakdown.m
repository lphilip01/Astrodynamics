function out = qns_perturbation_breakdown(t, x, params)
% qns_perturbation_breakdown
%
% Returns per-perturbation RTN accelerations and QNS element rates.
%
% State:
%   x = [a; ex; ey; inc; RAAN; u]
%
% Optional ephemeris input is passed THROUGH PARAMS:
%
%   params.ephem can be one of:
%
%   (A) Precomputed tables for interpolation:
%       params.ephem.t      : Nt x 1 time vector [s]
%       params.ephem.rSun   : Nt x 3 Sun position wrt Earth [km]
%       params.ephem.rMoon  : Nt x 3 Moon position wrt Earth [km]
%       params.ephem.method : interpolation method (optional, default 'linear')
%
%   (B) Direct vectors at current time:
%       params.ephem.rSunNow  : 3x1 or 1x3
%       params.ephem.rMoonNow : 3x1 or 1x3
%
% If params.ephem is absent or incomplete, planetEphemeris is used.
%
% Output struct fields:
%   out.accel.J2, out.accel.SRP, out.accel.Moon, out.accel.Sun
%   out.rates.J2, out.rates.SRP, out.rates.Moon, out.rates.Sun, out.rates.total
%

% Unpack state
a    = x(1);
inc  = x(4);
RAAN = x(5);
u    = x(6);

% Parameters
mu     = params.mu;
RE     = params.RE;
J2     = params.J2;
muMoon = params.muMoon;
muSun  = params.muSun;
CR     = params.CR;
As     = params.As;
m      = params.m;
S      = params.S;
c      = params.c;

if isfield(params,'ephemModel')
    ephModel = params.ephemModel;
else
    ephModel = '421';
end

if isfield(params,'useShadow')
    useShadow = params.useShadow;
else
    useShadow = true;
end

jd = params.jd0 + t/86400;

% Mean motion
n = sqrt(mu/a^3);

% Geometry
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

Nhat = [ sO*si;
        -cO*si;
         ci ];

r_sat = r * Rhat;

sin_i = sin(inc);
if abs(sin_i) < 1e-10
    sin_i = sign(sin_i + eps) * 1e-10;
end

%% ------------------------------------------------------------------------
% Get Sun / Moon ephemerides from params.ephem if available
%% ------------------------------------------------------------------------
r_sun  = [];
r_moon = [];

if isfield(params, 'ephem')
    ephem = params.ephem;

    % Case A: direct vectors supplied at this call
    if isfield(ephem, 'rSunNow') && ~isempty(ephem.rSunNow)
        r_sun = ephem.rSunNow(:);
    end
    if isfield(ephem, 'rMoonNow') && ~isempty(ephem.rMoonNow)
        r_moon = ephem.rMoonNow(:);
    end

    % Case B: interpolate from precomputed tables
    if isempty(r_sun) && isfield(ephem, 't') && isfield(ephem, 'rSun')
        if isfield(ephem, 'method') && ~isempty(ephem.method)
            interpMethod = ephem.method;
        else
            interpMethod = 'linear';
        end
        r_sun = interp1(ephem.t, ephem.rSun, t, interpMethod, 'extrap').';
    end

    if isempty(r_moon) && isfield(ephem, 't') && isfield(ephem, 'rMoon')
        if isfield(ephem, 'method') && ~isempty(ephem.method)
            interpMethod = ephem.method;
        else
            interpMethod = 'linear';
        end
        r_moon = interp1(ephem.t, ephem.rMoon, t, interpMethod, 'extrap').';
    end
end

% Fall back to planetEphemeris if still missing
if isempty(r_sun)
    r_sun = planetEphemeris(jd, 'Earth', 'Sun', ephModel);
    r_sun = r_sun(:);
end

if isempty(r_moon)
    r_moon = planetEphemeris(jd, 'Earth', 'Moon', ephModel);
    r_moon = r_moon(:);
end

%% ---------- J2 ----------
facJ2 = -(3/2) * J2 * mu * RE^2 / a^4;
AR_J2 = facJ2 * (1 - 3*si^2*su^2);
AT_J2 = facJ2 * (si^2 * sin(2*u));
AN_J2 = facJ2 * (sin(2*inc) * su);

%% ---------- SRP ----------
if useShadow
    nu_shadow = los_qns(r_sat, r_sun, RE);
else
    nu_shadow = 1;
end
if m==0
pSR=0;    
else
pSR = nu_shadow * (S/c) * CR * As / m / 1000;  % km/s^2
end
s_hat = r_sun / norm(r_sun);

ur_srp = dot(s_hat, Rhat);
us_srp = dot(s_hat, That);
uw_srp = dot(s_hat, Nhat);

AR_SRP = -pSR * ur_srp;
AT_SRP = -pSR * us_srp;
AN_SRP = -pSR * uw_srp;

%% ---------- Moon ----------
rho_moon = r_moon - r_sat;
a_moon_eci = muMoon * (rho_moon/norm(rho_moon)^3 - r_moon/norm(r_moon)^3);

AR_Moon = dot(a_moon_eci, Rhat);
AT_Moon = dot(a_moon_eci, That);
AN_Moon = dot(a_moon_eci, Nhat);

%% ---------- Sun gravity ----------
rho_sun = r_sun - r_sat;

if muSun==0
a_sun_eci=zeros(1,length(Rhat));
else
a_sun_eci = muSun * (rho_sun/norm(rho_sun)^3 - r_sun/norm(r_sun)^3);
end

AR_Sun = dot(a_sun_eci, Rhat);
AT_Sun = dot(a_sun_eci, That);
AN_Sun = dot(a_sun_eci, Nhat);

%% ---------- RTN accel -> QNS rates ----------
    % Perturbation-only rates (no n in udot)
function q = accel_to_qns_perturb(AR,AT,AN)
    adot     = (2/n) * AT;
    udot     = -(2/(n*a))*AR - (cos(inc)/(n*a*sin_i))*sin(u)*AN;  % no n
    exdot    = (1/(n*a))*( sin(u)*AR + 2*cos(u)*AT );
    eydot    = (1/(n*a))*(-cos(u)*AR + 2*sin(u)*AT );
    idot     = (1/(n*a))* cos(u)*AN;
    Omegadot = (1/(n*a*sin_i))* sin(u)*AN;
    q = [adot; exdot; eydot; idot; Omegadot; udot];
end

rates_J2   = accel_to_qns_perturb(AR_J2,  AT_J2,  AN_J2);
rates_SRP  = accel_to_qns_perturb(AR_SRP, AT_SRP, AN_SRP);
rates_Moon = accel_to_qns_perturb(AR_Moon,AT_Moon,AN_Moon);
rates_Sun  = accel_to_qns_perturb(AR_Sun, AT_Sun, AN_Sun);

out.accel.J2   = [AR_J2;   AT_J2;   AN_J2];
out.accel.SRP  = [AR_SRP;  AT_SRP;  AN_SRP];
out.accel.Moon = [AR_Moon; AT_Moon; AN_Moon];
out.accel.Sun  = [AR_Sun;  AT_Sun;  AN_Sun];

out.rates.J2    = rates_J2;
out.rates.SRP   = rates_SRP;
out.rates.Moon  = rates_Moon;
out.rates.Sun   = rates_Sun;
out.rates.total = rates_J2 + rates_SRP + rates_Moon + rates_Sun;
out.rates.total(6) = out.rates.total(6) + n;  % n added exactly once

end