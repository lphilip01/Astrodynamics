function [opd_dot, aux] = opd_rate_from_roe(roe, roe_dot, chief, mode, target)
% opd_rate_from_roe
%
% Analytical OPD rate from ROE, ROE rates, and source geometry.
%
% Inputs:
%   roe      : Nx6 or 1x6
%              [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]
%
%   roe_dot  : Nx6 or 1x6
%              time derivative of ROE in same order
%
%   chief    : struct
%              Required fields:
%                chief.a       semimajor axis [km]
%                chief.u       chief mean argument of latitude [rad]
%                chief.a_dot   semimajor axis rate [km/s]
%                chief.u_dot   mean argument of latitude rate [rad/s]
%
%              Additional fields required for mode='radec':
%                chief.inc       inclination [rad]
%                chief.RAAN      RAAN [rad]
%                chief.inc_dot   inclination rate [rad/s]
%                chief.RAAN_dot  RAAN rate [rad/s]
%
%   mode     : 'phibeta' or 'radec'
%
%   target   : Nx2 or 1x2
%              if mode='phibeta': [phi0, beta] [rad]
%              if mode='radec'   : [RA, Dec] [rad]
%
% Outputs:
%   opd_dot  : Nx1 OPD rate [km/s]
%
%   aux      : struct with useful intermediate outputs
%              aux.opd        instantaneous OPD [km]
%              aux.dr_rtn     RTN relative position [km]
%              aux.dr_rtn_dot RTN relative velocity-like derivative [km/s]
%              aux.s_rtn      source unit vector in RTN
%              aux.s_rtn_dot  time derivative of source unit vector in RTN [1/s]
%
% Notes:
%   - This function uses the analytic chain rule.
%   - For mode='radec', the source is inertially fixed but rotates in RTN
%     because the chief orbit frame changes.
%

% ---------------------------
% Input shaping
% ---------------------------
if isvector(roe),     roe     = roe(:).'; end
if isvector(roe_dot), roe_dot = roe_dot(:).'; end
if isvector(target),  target  = target(:).'; end

N = size(roe,1);

if size(roe,2) ~= 6 || size(roe_dot,2) ~= 6
    error('roe and roe_dot must be Nx6 or 1x6.');
end

if size(roe_dot,1) == 1 && N > 1
    roe_dot = repmat(roe_dot, N, 1);
elseif size(roe_dot,1) ~= N
    error('roe and roe_dot must have same number of rows.');
end

if size(target,2) ~= 2
    error('target must be Nx2 or 1x2.');
end

if size(target,1) == 1 && N > 1
    target = repmat(target, N, 1);
elseif size(target,1) ~= N
    error('target must be 1x2 or have same number of rows as roe.');
end

expandField = @(x) local_expand_field(x, N);

a     = expandField(chief.a);
u     = expandField(chief.u);
a_dot = expandField(chief.a_dot);
u_dot = expandField(chief.u_dot);

% ROE and rates
da   = roe(:,1);
dl   = roe(:,2);
dex  = roe(:,3);
dey  = roe(:,4);
dix  = roe(:,5);
diy  = roe(:,6);

da_d   = roe_dot(:,1);
dl_d   = roe_dot(:,2);
dex_d  = roe_dot(:,3);
dey_d  = roe_dot(:,4);
dix_d  = roe_dot(:,5);
diy_d  = roe_dot(:,6);

cu = cos(u);
su = sin(u);

% ---------------------------
% RTN relative position from ROE
% ---------------------------
FR = da - cu.*dex - su.*dey;
FT = dl + 2*su.*dex - 2*cu.*dey;
FN = su.*dix - cu.*diy;

dR = a .* FR;
dT = a .* FT;
dN = a .* FN;

dr_rtn = [dR, dT, dN];

% ---------------------------
% Analytic derivative of RTN relative position
% ---------------------------
FR_dot = da_d ...
       + (su.*u_dot).*dex - cu.*dex_d ...
       - (cu.*u_dot).*dey - su.*dey_d;

FT_dot = dl_d ...
       + 2*(cu.*u_dot).*dex + 2*su.*dex_d ...
       + 2*(su.*u_dot).*dey - 2*cu.*dey_d;

FN_dot = (cu.*u_dot).*dix + su.*dix_d ...
       + (su.*u_dot).*diy - cu.*diy_d;

dR_dot = a_dot .* FR + a .* FR_dot;
dT_dot = a_dot .* FT + a .* FT_dot;
dN_dot = a_dot .* FN + a .* FN_dot;

dr_rtn_dot = [dR_dot, dT_dot, dN_dot];

% ---------------------------
% Source direction and its derivative in RTN
% ---------------------------
switch lower(mode)
    case 'phibeta'
        phi0 = target(:,1);
        beta = target(:,2);

        cb = cos(beta);
        sb = sin(beta);
        psi = phi0 - u;

        cpsi = cos(psi);
        spsi = sin(psi);

        % Source direction in RTN
        s_rtn = [cb.*cpsi, cb.*spsi, sb];

        % Since psi = phi0 - u, psi_dot = -u_dot
        % d/dt cos(psi) = +sin(psi)*u_dot
        % d/dt sin(psi) = -cos(psi)*u_dot
        s_rtn_dot = [cb.*spsi.*u_dot, ...
                    -cb.*cpsi.*u_dot, ...
                     zeros(N,1)];

    case 'radec'
        % Need chief orbit orientation and rates
        if ~isfield(chief,'inc') || ~isfield(chief,'RAAN') || ...
           ~isfield(chief,'inc_dot') || ~isfield(chief,'RAAN_dot')
            error(['For mode=''radec'', chief must contain fields: ', ...
                   'inc, RAAN, inc_dot, RAAN_dot']);
        end

        inc      = expandField(chief.inc);
        RAAN     = expandField(chief.RAAN);
        inc_dot  = expandField(chief.inc_dot);
        RAAN_dot = expandField(chief.RAAN_dot);

        ra  = target(:,1);
        dec = target(:,2);

        % Inertial unit source direction
        s_eci = [cos(dec).*cos(ra), ...
                 cos(dec).*sin(ra), ...
                 sin(dec)];

        % RTN basis vectors
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

        % Time derivatives of RTN basis vectors
        % Rhat = [cO cu - sO su ci;
        %         sO cu + cO su ci;
        %         su si]
        %
        % Differentiate w.r.t. RAAN, inc, u

        Rhat_dot = [ ...
            (-sO.*RAAN_dot).*cu + cO.*(-su.*u_dot) ...
            - ( cO.*RAAN_dot).*su.*ci - sO.*(cu.*u_dot).*ci + sO.*su.*si.*inc_dot, ...
            ( cO.*RAAN_dot).*cu + sO.*(-su.*u_dot) ...
            + (-sO.*RAAN_dot).*su.*ci + cO.*(cu.*u_dot).*ci - cO.*su.*si.*inc_dot, ...
            ( cu.*u_dot).*si + su.*ci.*inc_dot ];

        That_dot = [ ...
            -((-sO.*RAAN_dot).*su + cO.*(cu.*u_dot)) ...
            -(( cO.*RAAN_dot).*cu.*ci - sO.*(su.*u_dot).*ci - sO.*cu.*si.*inc_dot), ...
            -(( cO.*RAAN_dot).*su + sO.*(cu.*u_dot)) ...
            +((-sO.*RAAN_dot).*cu.*ci + cO.*(-su.*u_dot).*ci - cO.*cu.*si.*inc_dot), ...
            (-su.*u_dot).*si + cu.*ci.*inc_dot ];

        Nhat_dot = [ ...
            cO.*RAAN_dot.*si + sO.*ci.*inc_dot, ...
            sO.*RAAN_dot.*si - cO.*ci.*inc_dot, ...
            -si.*inc_dot ];

        % Project source into RTN
        s_rtn = [sum(s_eci .* Rhat, 2), ...
                 sum(s_eci .* That, 2), ...
                 sum(s_eci .* Nhat, 2)];

        s_rtn_dot = [sum(s_eci .* Rhat_dot, 2), ...
                     sum(s_eci .* That_dot, 2), ...
                     sum(s_eci .* Nhat_dot, 2)];

    otherwise
        error('Unknown mode. Use ''phibeta'' or ''radec''.');
end

% ---------------------------
% OPD and OPD rate
% ---------------------------
opd = sum(dr_rtn .* s_rtn, 2);
opd_dot = sum(dr_rtn_dot .* s_rtn, 2) + sum(dr_rtn .* s_rtn_dot, 2);

% Diagnostics
aux.opd = opd;
aux.dr_rtn = dr_rtn;
aux.dr_rtn_dot = dr_rtn_dot;
aux.s_rtn = s_rtn;
aux.s_rtn_dot = s_rtn_dot;

end

% -------------------------------------------------------------------------
function x = local_expand_field(x, N)
if isscalar(x)
    x = repmat(x, N, 1);
else
    x = x(:);
    if length(x) ~= N
        error('Chief field must be scalar or length N.');
    end
end
end