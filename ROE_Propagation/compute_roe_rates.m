function roe_dot = compute_roe_rates(roe, orbital_state, perturbations)
%COMPUTE_ROE_RATES  Time derivatives of absolute ROE from Keplerian + perturbations.
%
% Computes instantaneous ROE rates under:
%   1. Keplerian secular drift (active)
%   2. Solar Radiation Pressure via GVE (architecture placeholder)
%
% KEPLERIAN DYNAMICS:
%   Under two-body gravity, the quasi-nonsingular ROE are first integrals
%   of the relative motion except for the secular dlambda drift:
%     dlambda_dot = -(3/2) * n * da
%   This arises because a deputy at different SMA (da != 0) completes
%   each orbit in a slightly different period (T = 2*pi/n, n ~ a^(-3/2)),
%   causing the along-track separation to grow linearly with time.
%
% SRP EXTENSION ARCHITECTURE (Curtis Ch. 12):
%   Solar radiation pressure exerts an acceleration in the RTN frame:
%     a_SRP = -nu * P_SR * C_R * (A/m)  [m/s^2]
%   where nu = shadow function (0 in eclipse, 1 in full sunlight).
%
%   The differential acceleration between spacecraft with different A/m ratios:
%     delta_a_SRP_RTN = [f_R; f_T; f_N]  (differential RTN acceleration)
%
%   Maps to ROE rates via Gauss Variational Equations (Curtis Eq. 12.84-12.89):
%     da_dot     = (2/n) * f_T                                     [Eq. 12.84]
%     dlambda_dot= -(3/2)*n*da + (2/n*e)*... + ...               [complete form]
%     dex_dot    = (1/na)[2*f_T*cos_nu + f_R*sin_nu]             [Eq. 12.87]
%     dey_dot    = (1/na)[2*f_T*sin_nu - f_R*cos_nu]
%     dix_dot    = (f_N/na) * cos(u)
%     diy_dot    = (f_N/na) * sin(u)
%   (Near-circular orbit approximations: e~0, u ~ nu + omega)
%
% Inputs:
%   roe            - Current absolute ROE [m], 6x1: [da; dlambda; dex; dey; dix; diy]
%   orbital_state  - Chief orbit state struct:
%                    .n     : mean motion [rad/s]
%                    .a     : semi-major axis [m]
%                    .e     : eccentricity [-]
%                    .inc   : inclination [rad]
%                    .omega : argument of perigee [rad]
%                    .u_c   : mean argument of latitude [rad]
%   perturbations  - Perturbation configuration struct:
%                    .srp_enabled      : logical toggle
%                    .C_R              : radiation pressure coefficient (1.0-2.0)
%                    .AmR_chief        : chief area-to-mass [m^2/kg]
%                    .AmR_deputy       : deputy area-to-mass [m^2/kg]
%                    .sun_direction_ECI: unit vector to Sun in ECI frame [3x1]
%
% Outputs:
%   roe_dot - ROE time derivatives [m/s], 6x1
%
% Physical constants (used in SRP module when implemented):
%   P_SR = 4.56e-6 N/m^2   (solar radiation pressure at 1 AU)
%   AU   = 1.496e11 m       (1 astronomical unit)

    % --- Initialize rates to zero ---
    roe_dot = zeros(6, 1);

    % --- Keplerian secular drift: dlambda driven by da ---
    % This is the only nonzero Keplerian ROE rate.
    % Physical meaning: deputy with da != 0 has mean motion n_d = n*(1 - 3/2*da/a_c)
    % giving along-track drift rate = a_c * (n - n_d) = (3/2)*n*da [m/s]
    roe_dot(2) = -(3/2) * orbital_state.n * roe(1);   % dlambda_dot [m/s]

    % --- SRP perturbation placeholder ---
    % Complete this block when adding the SRP perturbation module.
    % See: Curtis "Orbital Mechanics for Engineering Students", Chapter 12.
    if perturbations.srp_enabled

        % Physical constants
        P_SR = 4.56e-6;    % solar radiation pressure at 1 AU [N/m^2]

        % Step 1: Compute differential SRP acceleration magnitude
        % (deputy minus chief, since we model relative motion)
        % a_SRP = -nu * P_SR * C_R * (A/m)   [m/s^2, scalar magnitude]
        %   nu = shadow function (TODO: compute from sun_direction_ECI + orbit)
        nu = 1.0;   % assume full sunlight (placeholder)
        a_SRP_chief  = -nu * P_SR * perturbations.C_R * perturbations.AmR_chief;
        a_SRP_deputy = -nu * P_SR * perturbations.C_R * perturbations.AmR_deputy;
        delta_a_SRP  = a_SRP_deputy - a_SRP_chief;   % differential [m/s^2]

        % Step 2: Resolve differential SRP into RTN components
        % TODO: Rotate sun_direction_ECI into RTN frame using orbit geometry
        % Requires: RAAN, inclination, argument of latitude
        % perturbations.srp_accel_RTN = [f_R; f_T; f_N];   % [m/s^2]
        % (placeholder: assume sun is in T direction for now)
        f_R = 0;
        f_T = delta_a_SRP;
        f_N = 0;

        % Step 3: Apply Gauss Variational Equations (near-circular form)
        % Reference: Curtis (2014), Eqs. 12.84-12.89
        n   = orbital_state.n;
        a   = orbital_state.a;
        u_c = orbital_state.u_c;
        % TODO: Uncomment and complete when SRP module is ready:
        % roe_dot(1) = roe_dot(1) + (2/n) * f_T;                           % da_dot
        % roe_dot(2) = roe_dot(2) + (2/n) * f_R * sin(u_c) - ...;         % dlambda_dot (complete form)
        % roe_dot(3) = roe_dot(3) + (1/(n*a)) * (2*f_T*cos(u_c) + f_R*sin(u_c));  % dex_dot
        % roe_dot(4) = roe_dot(4) + (1/(n*a)) * (2*f_T*sin(u_c) - f_R*cos(u_c));  % dey_dot
        % roe_dot(5) = roe_dot(5) + (f_N/(n*a)) * cos(u_c);               % dix_dot
        % roe_dot(6) = roe_dot(6) + (f_N/(n*a)) * sin(u_c);               % diy_dot

        warning('ROE:SRPNotImplemented', ...
            'SRP perturbation module architecture active but rates not yet computed.');
    end
end
