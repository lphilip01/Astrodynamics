function [t_vec, roe_traj, u_c_vec] = propagate_roe(roe0, chief, N_orbits, pts_per_orbit)
%PROPAGATE_ROE  Keplerian propagation of quasi-nonsingular absolute ROE.
%
% Computes the time evolution of the absolute ROE state vector [m] under
% pure Keplerian dynamics. The quasi-nonsingular ROE (D'Amico & Montenbruck
% 2006) are nearly all constants of motion under Keplerian dynamics, with
% one exception: dlambda drifts secularly if da != 0.
%
% Physical interpretation of Keplerian dynamics:
%   - da != 0: deputy orbits at different SMA => different mean motion =>
%     along-track separation grows linearly with time (secular drift).
%   - da  = 0: all ROE constant => bounded, periodic relative motion.
%   - dex, dey: set shape and orientation of the 2:1 CW ellipse in R-T plane.
%   - dix, diy: set amplitude and phase of cross-track oscillation.
%
% Propagation rules (absolute ROE in meters):
%   da(t)      = da(0)                          [constant]
%   dlambda(t) = dlambda(0) - (3/2)*n*da(0)*t  [secular drift]
%   dex(t)     = dex(0)                         [constant]
%   dey(t)     = dey(0)                         [constant]
%   dix(t)     = dix(0)                         [constant]
%   diy(t)     = diy(0)                         [constant]
%
% Architecture note:
%   This function calls compute_roe_rates() for perturbation extensibility.
%   Currently only the Keplerian secular drift is active. SRP perturbations
%   (via Gauss Variational Equations) can be enabled via the perturbations
%   struct when that module is implemented.
%
% Inputs:
%   roe0          - Initial ROE struct (all fields in meters):
%                   .da, .dlambda, .dex, .dey, .dix, .diy
%   chief         - Chief orbit struct:
%                   .a [m], .mu [m^3/s^2], .M0 [deg], .omega [deg]
%   N_orbits      - Number of orbits to propagate [scalar]
%   pts_per_orbit - Time resolution (minimum 360 pts/orbit for smooth trajectories)
%
% Outputs:
%   t_vec    - Time vector [s],                1 x N_pts
%   roe_traj - ROE time history [m],           6 x N_pts
%              Rows: [da; dlambda; dex; dey; dix; diy]
%   u_c_vec  - Chief mean argument of latitude [rad], 1 x N_pts
%
% References:
%   D'Amico & Montenbruck (2006), "Proximity Operations of Formation-Flying
%   Spacecraft Using an Eccentricity/Inclination Vector Separation"

    % --- Input validation ---
    if pts_per_orbit < 360
        pts_per_orbit = 360;
    end
    if N_orbits <= 0
        error('N_orbits must be positive.');
    end

    % --- Chief orbit derived quantities ---
    n     = sqrt(chief.mu / chief.a^3);   % mean motion [rad/s]
    T_orb = 2*pi / n;                      % orbital period [s]

    % --- Time vector ---
    N_pts = max(2, round(N_orbits * pts_per_orbit));
    t_vec = linspace(0, N_orbits * T_orb, N_pts);   % [s]

    % --- Chief mean argument of latitude ---
    % u_c = omega + M  (circular orbit assumption: nu ~ M for e=0)
    u_c0    = deg2rad(chief.omega + chief.M0);   % initial value [rad]
    u_c_vec = u_c0 + n * t_vec;                   % advances linearly [rad]

    % --- Unpack initial absolute ROE [m] ---
    da0      = roe0.da;
    dlambda0 = roe0.dlambda;
    dex0     = roe0.dex;
    dey0     = roe0.dey;
    dix0     = roe0.dix;
    diy0     = roe0.diy;

    % --- Keplerian propagation (vectorized over all time steps) ---
    % All components constant except dlambda, which drifts at rate -(3/2)*n*da
    roe_traj        = zeros(6, N_pts);
    roe_traj(1, :)  = da0;                                     % da [m]  constant
    roe_traj(2, :)  = dlambda0 - (3/2) * n * da0 * t_vec;     % dlambda [m]  drifts
    roe_traj(3, :)  = dex0;                                    % dex [m] constant
    roe_traj(4, :)  = dey0;                                    % dey [m] constant
    roe_traj(5, :)  = dix0;                                    % dix [m] constant
    roe_traj(6, :)  = diy0;                                    % diy [m] constant

    % --- Perturbation integration (placeholder for future SRP extension) ---
    % When perturbations.srp_enabled = true, this block will integrate
    % differential SRP via the Gauss Variational Equations (Curtis Ch. 12).
    % The Euler integration loop below is the architecture hook:
    %
    % perturbations.srp_enabled = false;
    % orbital_state.n     = n;
    % orbital_state.a     = chief.a;
    % orbital_state.e     = chief.e;
    % orbital_state.inc   = deg2rad(chief.i);
    % orbital_state.omega = deg2rad(chief.omega);
    %
    % if perturbations.srp_enabled
    %     roe_srp = [da0; dlambda0; dex0; dey0; dix0; diy0];
    %     for k = 2:N_pts
    %         dt = t_vec(k) - t_vec(k-1);
    %         orbital_state.u_c = u_c_vec(k-1);
    %         roe_dot = compute_roe_rates(roe_srp, orbital_state, perturbations);
    %         roe_srp = roe_srp + roe_dot * dt;
    %         roe_traj(:, k) = roe_srp;
    %     end
    % end
end
