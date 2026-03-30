function roe_dot = qnsrates_to_roerates(xc, xd, xcdot, xddot)
% qnsrates_to_roerates
%
% Convert chief/deputy QNS state rates into ROE rates.
%
% Inputs:
%   xc    : 1x6 or 6x1 chief QNS state  [a, ex, ey, i, RAAN, u]
%   xd    : 1x6 or 6x1 deputy QNS state [a, ex, ey, i, RAAN, u]
%   xcdot : 1x6 or 6x1 chief QNS rate
%   xddot : 1x6 or 6x1 deputy QNS rate
%
% Output:
%   roe_dot : 1x6
%             [delta_a_dot, delta_lambda_dot, delta_ex_dot, ...
%              delta_ey_dot, delta_ix_dot, delta_iy_dot]
%
% Definitions:
%   delta_a   = (ad - ac)/ac
%   delta_lam = (ud - uc) + (RAANd - RAANc)*cos(ic)
%   delta_ex  = exd - exc
%   delta_ey  = eyd - eyc
%   delta_ix  = id - ic
%   delta_iy  = (RAANd - RAANc)*sin(ic)
%

xc    = xc(:).';
xd    = xd(:).';
xcdot = xcdot(:).';
xddot = xddot(:).';

% States
ac    = xc(1);
exc   = xc(2); %#ok<NASGU>
eyc   = xc(3); %#ok<NASGU>
ic    = xc(4);
RAc   = xc(5);
uc    = xc(6);

ad    = xd(1);
exd   = xd(2); %#ok<NASGU>
eyd   = xd(3); %#ok<NASGU>
id    = xd(4);
RAd   = xd(5);
ud    = xd(6);

% Rates
acdot  = xcdot(1);
excdot = xcdot(2);
eycdot = xcdot(3);
icdot  = xcdot(4);
RAcdot = xcdot(5);
ucdot  = xcdot(6);

addot  = xddot(1);
exddot = xddot(2);
eyddot = xddot(3);
iddot  = xddot(4);
RAddot = xddot(5);
uddot  = xddot(6);

% Wrapped relative angles
dRA = wrapToPi_local(RAd - RAc);
du  = wrapToPi_local(ud  - uc); %#ok<NASGU> % not directly needed after derivative

% Current ROE state components needed in derivatives
delta_a = (ad - ac)/ac;

% ROE rates
delta_a_dot = (addot - acdot)/ac - delta_a*(acdot/ac);

delta_lambda_dot = (uddot - ucdot) ...
                 + (RAddot - RAcdot)*cos(ic) ...
                 - dRA*sin(ic)*icdot;

delta_ex_dot = exddot - excdot;
delta_ey_dot = eyddot - eycdot;

delta_ix_dot = iddot - icdot;

delta_iy_dot = (RAddot - RAcdot)*sin(ic) ...
             + dRA*cos(ic)*icdot;

roe_dot = [delta_a_dot, delta_lambda_dot, delta_ex_dot, ...
           delta_ey_dot, delta_ix_dot, delta_iy_dot];
end

% -------------------------------------------------------------------------
function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end