function [opd, s_rtn] = opd_from_rtn(dr_rtn, mode, target, chief)
% opd_from_rtn
%
% Computes optical path delay (OPD) from a chief-centered RTN baseline.
%
% Inputs:
%   dr_rtn : Nx3 or 1x3 relative position in chief RTN frame [km]
%            dr_rtn = [dR, dT, dN]
%
%   mode   : string
%            'radec'    -> target = [RA, Dec] in radians
%            'phibeta'  -> target = [phi0, beta] in radians
%
%   target : Nx2 or 1x2
%            If mode = 'radec':
%                target = [RA, Dec] [rad]
%            If mode = 'phibeta':
%                target = [phi0, beta] [rad]
%
%   chief  : struct with chief geometry/orbit information
%
%            Required for mode = 'radec':
%               chief.inc   : chief inclination [rad], Nx1 or scalar
%               chief.RAAN  : chief RAAN [rad], Nx1 or scalar
%               chief.u     : chief mean argument of latitude [rad], Nx1 or scalar
%
%            Required for mode = 'phibeta':
%               chief.u     : chief mean argument of latitude [rad], Nx1 or scalar
%
% Outputs:
%   opd    : Nx1 optical path delay [km]
%   s_rtn  : Nx3 unit target direction expressed in RTN
%
% Notes:
%   OPD = dr_rtn · s_rtn
%
%   For 'radec', the target is assumed fixed in inertial space, with
%   unit vector
%       s_eci = [cos(dec)cos(ra); cos(dec)sin(ra); sin(dec)].
%
%   For 'phibeta', the RTN target direction is defined as
%       s_rtn = [cos(beta)cos(phi0-u);
%                cos(beta)sin(phi0-u);
%                sin(beta)].
%

% --- Ensure row-wise arrays ---
if isvector(dr_rtn), dr_rtn = dr_rtn(:).'; end
if isvector(target), target = target(:).'; end

N = size(dr_rtn,1);

if size(dr_rtn,2) ~= 3
    error('dr_rtn must be Nx3 or 1x3.');
end

if size(target,2) ~= 2
    error('target must be Nx2 or 1x2.');
end

if size(target,1) == 1
    target = repmat(target, N, 1);
elseif size(target,1) ~= N
    error('target must be 1x2 or have same number of rows as dr_rtn.');
end

% Helper to expand chief fields
expandField = @(x) local_expand_field(x, N);

switch lower(mode)
    case 'radec'
        % Required chief fields
        if ~isfield(chief,'inc') || ~isfield(chief,'RAAN') || ~isfield(chief,'u')
            error('For mode=''radec'', chief must contain fields inc, RAAN, and u.');
        end

        inc  = expandField(chief.inc);
        RAAN = expandField(chief.RAAN);
        u    = expandField(chief.u);

        ra  = target(:,1);
        dec = target(:,2);

        % Inertial target direction
        s_eci = [cos(dec).*cos(ra), ...
                 cos(dec).*sin(ra), ...
                 sin(dec)];

        % RTN basis from chief orbit
        cO = cos(RAAN); sO = sin(RAAN);
        ci = cos(inc);  si = sin(inc);
        cu = cos(u);    su = sin(u);

        Rhat = [ cO.*cu - sO.*su.*ci, ...
                 sO.*cu + cO.*su.*ci, ...
                 su.*si ];

        That = [ -cO.*su - sO.*cu.*ci, ...
                 -sO.*su + cO.*cu.*ci, ...
                  cu.*si ];

        Nhat = [ sO.*si, ...
                -cO.*si, ...
                 ci ];

        % Project inertial target into RTN
        s_rtn = [sum(s_eci .* Rhat, 2), ...
                 sum(s_eci .* That, 2), ...
                 sum(s_eci .* Nhat, 2)];

    case 'phibeta'
        if ~isfield(chief,'u')
            error('For mode=''phibeta'', chief must contain field u.');
        end

        u = expandField(chief.u);

        phi0 = target(:,1);
        beta = target(:,2);

        s_rtn = [cos(beta).*cos(phi0 - u), ...
                 cos(beta).*sin(phi0 - u), ...
                 sin(beta)];

    otherwise
        error('Unknown mode. Use ''radec'' or ''phibeta''.');
end

% OPD = projection of baseline onto target direction
opd = sum(dr_rtn .* s_rtn, 2);

end

% -------------------------------------------------------------------------
function x = local_expand_field(x, N)
% Expand scalar/vector chief field to Nx1
if isscalar(x)
    x = repmat(x, N, 1);
else
    x = x(:);
    if length(x) ~= N
        error('Chief field dimensions must be scalar or length N.');
    end
end
end