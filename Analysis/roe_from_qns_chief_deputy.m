function roe = roe_from_qns_chief_deputy(xc, xd)
% roe_from_qns_chief_deputy
%
% Forms quasi-nonsingular relative orbital elements (ROE)
% from chief and deputy QNS states.
%
% Inputs:
%   xc : chief QNS state,  Nx6 or 6x1
%   xd : deputy QNS state, Nx6 or 6x1
%
% QNS state definition:
%   x = [a, ex, ey, inc, RAAN, u]
%
% where
%   a    = semimajor axis [km]
%   ex   = e*cos(omega)
%   ey   = e*sin(omega)
%   inc  = inclination [rad]
%   RAAN = right ascension of ascending node [rad]
%   u    = omega + M [rad]
%
% Output:
%   roe : Nx6 array of relative orbital elements
%
%   roe(:,1) = delta_a   = (ad - ac)/ac
%   roe(:,2) = delta_lam = (ud - uc) + (RAANd - RAANc)*cos(ic)
%   roe(:,3) = delta_ex  = exd - exc
%   roe(:,4) = delta_ey  = eyd - eyc
%   roe(:,5) = delta_ix  = id - ic
%   roe(:,6) = delta_iy  = (RAANd - RAANc)*sin(ic)
%
% Notes:
%   - Angle wrapping is handled so that relative angles remain in [-pi, pi].
%   - Chief and deputy inputs must have matching sizes.
%

% Ensure row-wise state history
if isvector(xc), xc = xc(:).'; end
if isvector(xd), xd = xd(:).'; end

if size(xc,2) ~= 6 || size(xd,2) ~= 6
    error('xc and xd must be Nx6 or 1x6 QNS state arrays.');
end

if size(xc,1) ~= size(xd,1)
    error('xc and xd must have the same number of rows.');
end

% Chief
ac    = xc(:,1);
exc   = xc(:,2);
eyc   = xc(:,3);
ic    = xc(:,4);
RAc   = xc(:,5);
uc    = xc(:,6);

% Deputy
ad    = xd(:,1);
exd   = xd(:,2);
eyd   = xd(:,3);
id    = xd(:,4);
RAd   = xd(:,5);
ud    = xd(:,6);

% Wrapped angle differences
dRA = wrapToPi_local(RAd - RAc);
du  = wrapToPi_local(ud  - uc);

% ROE
delta_a   = (ad - ac) ./ ac;
delta_lam = du + dRA .* cos(ic);
delta_ex  = exd - exc;
delta_ey  = eyd - eyc;
delta_ix  = id - ic;
delta_iy  = dRA .* sin(ic);

roe = [delta_a, delta_lam, delta_ex, delta_ey, delta_ix, delta_iy];

end

% -------------------------------------------------------------------------
function ang = wrapToPi_local(ang)
% Wrap angle to [-pi, pi]
ang = mod(ang + pi, 2*pi) - pi;
end