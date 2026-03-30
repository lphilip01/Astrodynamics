function dr_rtn = rtn_from_roe(roe, ac, uc)
% rtn_from_roe
%
% Maps near-circular quasi-nonsingular ROE into chief-centered RTN
% relative position.
%
% Inputs:
%   roe : Nx6 or 1x6 array of ROE
%         roe = [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]
%
%   ac  : chief semimajor axis [km]
%         scalar or Nx1
%
%   uc  : chief mean argument of latitude [rad]
%         scalar or Nx1
%
% Output:
%   dr_rtn : Nx3 array of relative position in RTN [km]
%            dr_rtn(:,1) = delta_r_R
%            dr_rtn(:,2) = delta_r_T
%            dr_rtn(:,3) = delta_r_N
%
% Mapping used:
%   delta_r_R / a = delta_a - cos(u)*delta_ex - sin(u)*delta_ey
%   delta_r_T / a = delta_lambda + 2*sin(u)*delta_ex - 2*cos(u)*delta_ey
%   delta_r_N / a = sin(u)*delta_ix - cos(u)*delta_iy
%
% Notes:
%   - Valid for near-circular chief orbit.
%   - This is the standard first-order ROE-to-RTN mapping.
%

% Ensure row-wise
if isvector(roe), roe = roe(:).'; end
if isscalar(ac), ac = repmat(ac, size(roe,1), 1); end
if isscalar(uc), uc = repmat(uc, size(roe,1), 1); end

if size(roe,2) ~= 6
    error('roe must be Nx6 or 1x6.');
end

if length(ac) ~= size(roe,1) || length(uc) ~= size(roe,1)
    error('ac and uc must be scalar or have same number of rows as roe.');
end

% Extract ROE
delta_a   = roe(:,1);
delta_lam = roe(:,2);
delta_ex  = roe(:,3);
delta_ey  = roe(:,4);
delta_ix  = roe(:,5);
delta_iy  = roe(:,6);

cu = cos(uc);
su = sin(uc);

% ROE -> RTN
dR = ac .* ( delta_a - cu .* delta_ex - su .* delta_ey );

dT = ac .* ( delta_lam + 2*su .* delta_ex - 2*cu .* delta_ey );

dN = ac .* ( su .* delta_ix - cu .* delta_iy );

dr_rtn = [dR, dT, dN];
end