function [delta_alpha_star, info] = STARI_science_orbit(delta_lambda, phi0_deg, beta_deg, ac)
% =========================================================================
% STARI_science_orbit.m
%
% Computes the target ROE for the STARI science orbit (Eq. 9 in paper).
%
% The science orbit ensures the relative position vector is always
% perpendicular to the star direction (interferometry requirement).
%
% Science orbit ROE (Eq. 9):
%   da     = 0
%   dlambda = delta_lambda (along-track separation, dimensionless)
%   dex    = 0
%   dey    = 0
%   dix    = delta_lambda * cos(phi0) / tan(beta)
%   diy    = delta_lambda * sin(phi0) / tan(beta)
%
% For phi0 = 90 deg, beta = 45 deg:
%   dix = 0
%   diy = delta_lambda  (frozen eccentricity condition satisfied)
%
% Inputs:
%   delta_lambda  along-track separation [dimensionless = meters / ac]
%                 e.g. for 100m separation at a=6878km: 100/6878e3
%   phi0_deg      star azimuth at ascending node [deg] (90 = frozen)
%   beta_deg      star orbit elevation angle [deg] (>41.81 for REQ.1)
%   ac            chief semi-major axis [m]
%
% Output:
%   delta_alpha_star  [6x1]  target ROE (dimensionless)
%   info              struct  baseline analysis
% =========================================================================

phi0 = phi0_deg * pi/180;
beta = beta_deg * pi/180;

da      = 0;
dlambda = delta_lambda;
dex     = 0;
dey     = 0;
dix     = delta_lambda * cos(phi0) / tan(beta);
diy     = delta_lambda * sin(phi0) / tan(beta);

delta_alpha_star = [da; dlambda; dex; dey; dix; diy];

% --- Baseline analysis (Table 2 in paper) ---
K = ac * abs(delta_lambda) / (2 * abs(sin(beta)));

B_min  = ac * abs(delta_lambda);
B_mean = K * (1 + abs(sin(beta)));
B_max  = 2 * K;
dB     = K * (1 - abs(sin(beta)));

ratio = B_max / B_min;   % must be < 1.5 for REQ.1

info.K         = K;
info.B_min     = B_min;
info.B_mean    = B_mean;
info.B_max     = B_max;
info.dB        = dB;
info.ratio     = ratio;
info.REQ1_ok   = (ratio < 1.5);
info.REQ1_beta_min = asin(1/1.5) * 180/pi;   % 41.81 deg

fprintf('Science Orbit ROE (phi0=%.0f deg, beta=%.0f deg):\n', phi0_deg, beta_deg);
fprintf('  a*da      = %.2f m\n', da*ac);
fprintf('  a*dlambda = %.2f m\n', dlambda*ac);
fprintf('  a*dex     = %.2f m\n', dex*ac);
fprintf('  a*dey     = %.2f m\n', dey*ac);
fprintf('  a*dix     = %.2f m\n', dix*ac);
fprintf('  a*diy     = %.2f m\n\n', diy*ac);
fprintf('Baseline analysis:\n');
fprintf('  B_min  = %.2f m\n',  B_min);
fprintf('  B_mean = %.2f m\n',  B_mean);
fprintf('  B_max  = %.2f m\n',  B_max);
fprintf('  B_max/B_min = %.3f  (REQ.1 limit: 1.5) -> %s\n\n', ...
        ratio, ifelse(info.REQ1_ok, 'PASS', 'FAIL'));

end

function out = ifelse(c,a,b)
if c; out=a; else; out=b; end
end
