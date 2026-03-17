function rtn = roe_to_rtn(roe_traj, u_c_vec)
%ROE_TO_RTN  Linear map from absolute ROE to RTN relative position.
%
% Applies the linearized state transition map from quasi-nonsingular ROE to
% relative position in the RTN (Radial-Tangential-Normal) frame, valid for
% near-circular chief orbits (e_c << 1).
%
% The mapping equations (D'Amico & Montenbruck 2006, Eq. using absolute ROE):
%
%   delta_r_R = da      - cos(u_c)*dex - sin(u_c)*dey
%   delta_r_T = dlambda + 2*sin(u_c)*dex - 2*cos(u_c)*dey
%   delta_r_N = sin(u_c)*dix - cos(u_c)*diy
%
% All quantities in meters. This is derived from the Hill-Clohessy-Wiltshire
% (HCW) equations by identifying the integration constants with the ROE.
%
% Physical interpretation:
%   RADIAL (R): constant offset da, plus oscillation from eccentricity vector.
%               Amplitude = |de|, phase = atan2(dey, dex).
%   TANGENTIAL (T): mean offset dlambda (drifts if da!=0), plus 2x the radial
%               eccentricity oscillation (the "2:1" ratio of the CW ellipse).
%   NORMAL (N): purely sinusoidal cross-track motion driven by inclination
%               vector di. Amplitude = |di|, phase = atan2(diy, dix) + pi/2.
%
% Note on the 2:1 ellipse: When da=0 and dlambda=0, only dex or dey nonzero:
%   rR = -cos(u)*dex, rT = 2*sin(u)*dex  => T-amplitude is 2x R-amplitude.
%   This is the classic Clohessy-Wiltshire "2:1 ellipse" in the R-T plane.
%
% Inputs:
%   roe_traj - Absolute ROE time history [m],  6 x N
%              Rows: [da; dlambda; dex; dey; dix; diy]
%   u_c_vec  - Chief mean argument of latitude [rad], 1 x N
%
% Outputs:
%   rtn      - RTN relative position [m], 3 x N
%              Row 1: Radial R (outward from Earth center)
%              Row 2: Tangential T (along velocity vector)
%              Row 3: Normal N (orbit normal, completes right-hand frame)

    % --- Extract ROE components [m] ---
    da      = roe_traj(1, :);   % differential SMA [m]
    dlambda = roe_traj(2, :);   % mean longitude separation [m]
    dex     = roe_traj(3, :);   % relative eccentricity vector, x [m]
    dey     = roe_traj(4, :);   % relative eccentricity vector, y [m]
    dix     = roe_traj(5, :);   % relative inclination vector, x [m]
    diy     = roe_traj(6, :);   % relative inclination vector, y [m]

    % --- Trig of chief mean argument of latitude ---
    cos_u = cos(u_c_vec);   % 1 x N
    sin_u = sin(u_c_vec);   % 1 x N

    % --- Linear RTN mapping (all in meters, fully vectorized) ---

    % Radial: constant offset + eccentricity-driven oscillation (amplitude |de|)
    rR = da - cos_u .* dex - sin_u .* dey;

    % Tangential: mean separation + 2x eccentricity oscillation
    % Factor of 2 is the HCW coupling (Clohessy-Wiltshire equations)
    rT = dlambda + 2*sin_u .* dex - 2*cos_u .* dey;

    % Normal: pure cross-track sinusoid from inclination vector
    % Amplitude = sqrt(dix^2 + diy^2), zero mean
    rN = sin_u .* dix - cos_u .* diy;

    % --- Assemble RTN position matrix ---
    rtn = [rR; rT; rN];   % 3 x N [m]
end
