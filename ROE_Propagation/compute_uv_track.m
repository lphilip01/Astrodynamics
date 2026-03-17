function [uv, opd, u_hat, v_hat, s_hat] = compute_uv_track(rtn, phi_0_deg, beta_deg)
%COMPUTE_UV_TRACK  Project inter-spacecraft baseline onto interferometric UV plane.
%
% For a target star in direction s_hat, decomposes the baseline vector B
% (= RTN relative position) into:
%   1. Component along s_hat: OPD (optical path difference) - drives fringe phase
%   2. Component perp to s_hat: B_perp - the interferometric "uv" coverage
%
% The UV plane is the plane perpendicular to the line-of-sight to the star.
% As the spacecraft orbit, the baseline sweeps different UV coordinates,
% providing spatial frequency coverage analogous to Earth-rotation aperture
% synthesis in radio interferometry.
%
% Star direction convention (angles defined in RTN frame):
%   phi_0 = azimuth measured from R-axis toward T-axis [deg]
%   beta  = elevation above the R-T plane [deg]
%   s_hat = [cos(b)*cos(p), cos(b)*sin(p), sin(b)]_RTN
%
% UV coordinate frame construction:
%   Reference: N_hat = [0, 0, 1]_RTN (or R_hat if star near zenith)
%   u_hat = normalize(N_hat x s_hat)    [roughly along -R for phi_0=90]
%   v_hat = s_hat x u_hat               [completes right-hand frame]
%
% STARI perpendicularity condition:
%   For maximum UV coverage efficiency, the orbital plane should be oriented
%   so the baseline sweeps maximally in the UV plane (minimising OPD variation
%   and maximising B_perp). This is achieved when the relative inclination
%   vector di is perpendicular to s_hat projected in the T-N plane.
%
% Inputs:
%   rtn       - RTN relative position (baseline) [m], 3 x N
%   phi_0_deg - Star azimuth from R toward T [deg], scalar
%   beta_deg  - Star elevation above R-T plane [deg], scalar
%
% Outputs:
%   uv    - UV plane coordinates [m], 2 x N (row 1=u, row 2=v)
%   opd   - Optical path difference B·s_hat [m], 1 x N
%   u_hat - U-axis unit vector in RTN frame (3x1)
%   v_hat - V-axis unit vector in RTN frame (3x1)
%   s_hat - Star line-of-sight unit vector in RTN frame (3x1)
%
% Reference: STARI paper (Rizza et al. 2026), interferometric baseline analysis

    % --- Star direction unit vector in RTN frame ---
    phi_0 = deg2rad(phi_0_deg);
    beta  = deg2rad(beta_deg);
    s_hat = [cos(beta)*cos(phi_0);    % R component
             cos(beta)*sin(phi_0);    % T component
             sin(beta)];              % N component

    % --- Build UV coordinate frame ---
    % Primary reference: N direction [0,0,1]
    % Fall back to R direction [1,0,0] if star is near zenith (s_hat ~ N_hat)
    ref = [0; 0; 1];
    if abs(dot(s_hat, ref)) > 0.98
        ref = [1; 0; 0];    % use R if nearly parallel to N
    end

    % u_hat: cross-product of reference and s_hat, normalized
    u_hat = cross(ref, s_hat);
    u_hat = u_hat / norm(u_hat);

    % v_hat: completes right-hand orthonormal frame {u_hat, v_hat, s_hat}
    v_hat = cross(s_hat, u_hat);
    v_hat = v_hat / norm(v_hat);

    % --- OPD: projection of baseline along line of sight [m] ---
    % OPD determines the fringe phase; must be << coherence length for fringes
    opd = s_hat' * rtn;    % 1 x N [m]

    % --- Perpendicular baseline: B_perp = B - (B·s_hat)*s_hat [m] ---
    % This is the component contributing to angular resolution (uv coverage)
    B_perp = rtn - s_hat * opd;    % 3 x N [m]

    % --- UV coordinates: projections of B_perp onto u_hat and v_hat ---
    uv = [u_hat' * B_perp;    % u = B_perp · u_hat,  1 x N [m]
          v_hat' * B_perp];   % v = B_perp · v_hat,  1 x N [m]
end
