function results = compute_opd_analytical_linear(star_ra, star_dec, ...
    orbit_params, formation_params, perturbation_params, ...
    perturbation_flags, D_max, t_eval)
% COMPUTE_OPD_ANALYTICAL  Analytical OPD growth rate and passive integration
% time for a combiner-on-ellipse interferometric formation.
%
% Computes the optical path difference (OPD) growth between a single 
% collector and the combiner spacecraft due to perturbation-driven drift 
% of the relative orbital elements (ROE) away from the nominal science 
% orbit. The OPD is the projection of the RTN relative position error 
% onto the target star direction vector.
%
% Each perturbation source (J2, SRP, lunisolar) can be independently 
% enabled or disabled via perturbation_flags to show the individual and 
% combined contributions to OPD growth.
%
% EQUATIONS IMPLEMENTED:
%   Star coordinate conversion: Hansen & Ireland (2020) Eqs. for (beta, phi0)
%   Nominal science orbit ROE: Rizza et al. (2026) Eq. 9
%   ROE-to-RTN mapping: Koenig, D'Amico & Lightsey (2023) Eq. 5
%   OPD harmonic decomposition: Rizza et al. (2026) Eqs. 7-8
%   J2 secular ROE rates: Koenig, Guffanti & D'Amico (2017) STM
%   SRP secular ROE rates: [PLACEHOLDER - to be implemented]
%   Lunisolar secular ROE rates: [PLACEHOLDER - to be implemented]
%
% INPUTS:
%   star_ra            - [1x1] Right ascension of target star [rad]
%   star_dec           - [1x1] Declination of target star [rad]
%   orbit_params       - struct with fields:
%       .a             - Combiner semimajor axis [km]
%       .inc           - Orbital inclination [rad]
%       .RAAN          - Right ascension of ascending node [rad]
%   formation_params   - struct with fields:
%       .delta_lambda  - Relative mean longitude [dimensionless]
%                        (a * delta_lambda gives along-track separation in km)
%       .B_max         - Maximum baseline [m] (used for display/validation only;
%                        delta_lambda is the primary design parameter)
%   perturbation_params - struct with fields:
%       .mu_earth      - Earth gravitational parameter [km^3/s^2]
%       .R_earth       - Mean equatorial radius of Earth [km]
%       .J2            - J2 zonal harmonic coefficient [dimensionless]
%       .P_srp         - Solar radiation pressure at 1 AU [N/m^2]
%       .delta_Bs      - Differential SRP coefficient (collector - combiner)
%                        [m^2/kg]  (C_R * A/m difference)
%       .sun_beta      - Sun elevation above orbital plane [rad]
%       .sun_lambda    - Sun in-plane angle from ascending node [rad]
%       .mu_sun        - Sun gravitational parameter [km^3/s^2]
%       .mu_moon       - Moon gravitational parameter [km^3/s^2]
%       .a_sun         - Sun semimajor axis (Earth-Sun distance) [km]
%       .a_moon        - Moon semimajor axis [km]
%       .moon_beta     - Moon elevation above orbital plane [rad]
%       .moon_lambda   - Moon in-plane angle from ascending node [rad]
%   perturbation_flags - struct with boolean fields:
%       .J2            - Enable J2 perturbation [true/false]
%       .SRP           - Enable SRP perturbation [true/false]
%       .lunisolar     - Enable lunisolar perturbation [true/false]
%   D_max              - [1x1] Delay line stroke [m]
%   t_eval             - [Mx1] Times at which to evaluate OPD envelope [s]
%
% OUTPUTS:
%   results            - struct with fields:
%       .T_passive         - Analytical passive integration time [s]
%       .opd_rate_dc       - DC component of OPD growth rate [m/s]
%       .opd_rate_1n       - 1st harmonic (1n) amplitude growth rate [m/s]
%       .opd_rate_2n       - 2nd harmonic (2n) amplitude growth rate [m/s]
%       .opd_rate_total    - Total peak OPD growth rate [m/s]
%       .roe_rates         - [6x1] Secular ROE error rates [1/s]
%                            [deps_da; deps_dlam; deps_dex; deps_dey; 
%                             deps_dix; deps_diy]
%       .roe_rates_J2      - [6x1] J2 contribution to ROE rates [1/s]
%       .roe_rates_SRP     - [6x1] SRP contribution to ROE rates [1/s]
%       .roe_rates_luni    - [6x1] Lunisolar contribution to ROE rates [1/s]
%       .opd_envelope      - [Mx1] Peak OPD envelope at each t_eval [m]
%       .opd_dc_hist       - [Mx1] DC OPD component history [m]
%       .opd_1n_hist       - [Mx1] 1st harmonic amplitude history [m]
%       .opd_2n_hist       - [Mx1] 2nd harmonic amplitude history [m]
%       .beta_star         - [1x1] Orbit elevation of target star [rad]
%       .phi0_star         - [1x1] Star azimuth at ascending node [rad]
%       .roe_nominal       - [6x1] Nominal science orbit ROE [dimensionless]
%       .s_hat_eci         - [3x1] Star unit vector in ECI [dimensionless]
%
% UNITS CONVENTION:
%   All orbital mechanics quantities use km and seconds internally.
%   OPD outputs are in meters.
%   SRP parameters (P_srp, delta_Bs) use SI units (N/m^2, m^2/kg).
%   The conversion factor 1e-3 km/m is applied when SRP accelerations 
%   (in m/s^2) enter expressions involving orbital quantities (in km).
%
% REFERENCES:
%   [1] Hansen & Ireland (2020), PASA, 37, e019
%   [2] Rizza et al. (2026), IEEE Aerospace Conference
%   [3] Koenig, D'Amico & Lightsey (2023), JGCD, 46(9), 1657-1670
%   [4] Koenig, Guffanti & D'Amico (2017), JGCD, 40(7), 1749-1768
%   [5] Ito (2024), A&A, 682, A38

% =========================================================================
%  STEP 1: Extract parameters
% =========================================================================

a     = orbit_params.a;        % [km]
inc   = orbit_params.inc;      % [rad]
RAAN  = orbit_params.RAAN;     % [rad]

dlam  = formation_params.delta_lambda;  % [dimensionless]

mu = 398600.4418;        % km^3/s^2    (GMAT default, EGM-96)
R_e  = 6378.137;           % km          (WGS-84 equatorial radius)
J2       = 1.08262668e-3;      % dimensionless (EGM-96)

% Mean motion [rad/s]
n = sqrt(mu / a^3);

% =========================================================================
%  STEP 2: Convert star (RA, Dec) to orbit-frame angles (beta, phi0)
%
%  Equations (E1)-(E3) from Hansen & Ireland (2020):
%    cos(beta) = cos(i)*sin(dec) + cos(dec)*sin(i)*sin(ra - RAAN)
%    sin(phi0) = [cos(dec)*cos(i)*sin(ra-RAAN) + sin(dec)*sin(i)] / sin(beta)
%    cos(phi0) = cos(dec)*cos(ra - RAAN) / sin(beta)
%
%  beta is the angle between the star direction and the orbital plane,
%  measured from the orbit normal. beta = 90 deg means the star is 
%  exactly in the orbital plane.
%
%  phi0 is the azimuthal angle of the star measured from the ascending
%  node within the orbital plane.
% =========================================================================

dRA = star_ra - RAAN;  % right ascension difference [rad]

cos_beta = cos(inc) * sin(star_dec) + ...
           cos(star_dec) * sin(inc) * sin(dRA);

% Clamp to [-1, 1] to avoid numerical issues with acos
cos_beta = max(-1, min(1, cos_beta));
beta_star = acos(cos_beta);   % [rad]
sin_beta  = sin(beta_star);

% Check observability: beta must be sufficiently far from 0 or pi
% (star must not be near the orbit normal)
beta_min_deg = 42;  % minimum beta for <50% baseline oscillation
if abs(beta_star) < deg2rad(beta_min_deg) || ...
   abs(beta_star) > deg2rad(180 - beta_min_deg)
    warning('compute_opd_analytical:betaConstraint', ...
        'Star elevation beta = %.1f deg violates the minimum %.1f deg constraint.', ...
        rad2deg(beta_star), beta_min_deg);
end

% Compute phi0 using atan2 for correct quadrant
sin_phi0 = (cos(star_dec) * cos(inc) * sin(dRA) + ...
            sin(star_dec) * sin(inc)) / sin_beta;
cos_phi0 = cos(star_dec) * cos(dRA) / sin_beta;

phi0_star = atan2(sin_phi0, cos_phi0);  % [rad]

% Star unit vector in ECI (J2000)
s_hat_eci = [cos(star_dec) * cos(star_ra); ...
             cos(star_dec) * sin(star_ra); ...
             sin(star_dec)];

% =========================================================================
%  STEP 3: Compute nominal science orbit ROE
%
%  Equation (E4) from Rizza et al. (2026) Eq. 9:
%    delta_alpha_nom = [0, dlam, 0, 0, dlam*cos(phi0)/tan(beta), 
%                       dlam*sin(phi0)/tan(beta)]
%
%  This is the ROE state that produces OPD = 0 for all time under 
%  CW dynamics. The formation is initialized here; any deviation from
%  these values is a ROE error that produces nonzero OPD.
% =========================================================================

tan_beta = tan(beta_star);

roe_nominal = [0; ...
               dlam; ...
               0; ...
               0; ...
               dlam * cos_phi0 / tan_beta; ...
               dlam * sin_phi0 / tan_beta];

% Along-track separation [m] for display
along_track_sep_m = a * abs(dlam) * 1e3;  % [m]

% Nominal relative inclination magnitude [m]
di_nom_m = a * abs(dlam) / abs(tan_beta) * 1e3;  % [m]

%% ========================================================================
%  STEP 3.5: Compute Geometric Baseline OPD (Keplerian, no perturbations)
%
%  Even with the nominal ROE and perfect Keplerian dynamics, if the star
%  direction does not exactly satisfy the zero-OPD condition, there is a
%  time-varying geometric OPD that oscillates at the orbital frequency.
%
%  For combiner-on-ellipse: The zero-OPD condition requires beta = 90 deg
%    (star in the orbital plane). If beta ≠ 90 deg, there is a geometric
%    OPD with amplitude ~ B_max * |sin(90deg - beta)| = B_max * |cos(beta)|.
%
%  The geometric OPD oscillates at frequencies n and 2n (from the RTN
%  relative position harmonics). We compute the peak amplitude here.
%  ========================================================================

% Compute RTN relative position at the nominal ROE using Koenig Eq. 5
% This gives the relative position as a function of u_c (mean arg of lat)
% We evaluate the OPD = delta_r · s_hat at a grid of u_c values to find peak

u_c_grid = linspace(0, 2*pi, 361);  % argument of latitude grid [rad]
opd_geometric_grid = zeros(size(u_c_grid));

for idx = 1:length(u_c_grid)
    u_c = u_c_grid(idx);
    
    % RTN relative position from nominal ROE (Koenig et al. 2023 Eq. 5)
    delta_rR = a * (roe_nominal(1) - cos(u_c)*roe_nominal(3) - sin(u_c)*roe_nominal(4));
    delta_rT = a * (roe_nominal(2) + 2*sin(u_c)*roe_nominal(3) - 2*cos(u_c)*roe_nominal(4));
    delta_rN = a * (sin(u_c)*roe_nominal(5) - cos(u_c)*roe_nominal(6));
    
    % Star direction in RTN (Rizza et al. 2026 Eq. 4)
    % Note: For combiner-on-ellipse, if the star were exactly in the
    % orbital plane, beta would be 90 deg and the star RTN vector would
    % rotate purely in the R-T plane with zero N component.
    s_R = cos_beta * cos(phi0_star - u_c);
    s_T = cos_beta * sin(phi0_star - u_c);
    s_N = sin_beta;
    
    % OPD = projection of relative position onto star direction [km]
    opd_geometric_grid(idx) = delta_rR * s_R + delta_rT * s_T + delta_rN * s_N;
end

% Convert to meters
opd_geometric_grid = opd_geometric_grid * 1e3;  % [m]

% Peak-to-peak geometric OPD
opd_geometric_pp = max(opd_geometric_grid) - min(opd_geometric_grid);  % [m]
opd_geometric_peak = max(abs(opd_geometric_grid));                     % [m]
opd_geometric_mean = mean(opd_geometric_grid);                         % [m]
opd_geometric_rms  = sqrt(mean(opd_geometric_grid.^2));               % [m]

fprintf('\n=== Geometric Baseline OPD (Keplerian) ===\n');
fprintf('Peak geometric OPD = %.4f m\n', opd_geometric_peak);
fprintf('Peak-to-peak geometric OPD = %.4f m\n', opd_geometric_pp);
fprintf('Mean geometric OPD = %.4f m\n', opd_geometric_mean);
fprintf('RMS geometric OPD = %.4f m\n', opd_geometric_rms);

if opd_geometric_peak > D_max
    warning('compute_opd_analytical:geometricOPDexceedsStroke', ...
        ['Geometric baseline OPD (%.2f m) exceeds delay line stroke (%.2f m).\n' ...
         'This star cannot be observed with this formation geometry.\n' ...
         'Consider:\n' ...
         '  1) Rotating the orbital plane to bring the star closer to in-plane\n' ...
         '  2) Using a longer delay line\n' ...
         '  3) Reducing the baseline B_max'], ...
        opd_geometric_peak, D_max);
end

% For the time-varying geometric OPD history, interpolate onto t_eval grid
% We assume u_c(t) = u_c_0 + n*t with u_c_0 = 0 at t=0 for simplicity
u_c_eval = mod(n * t_eval, 2*pi);  % wrap to [0, 2*pi]
opd_geometric_hist = interp1(u_c_grid, opd_geometric_grid, u_c_eval, 'linear', 'extrap');


% =========================================================================
%  STEP 4: Compute secular ROE error rates from each perturbation source
%
%  The ROE error is eps = delta_alpha(t) - delta_alpha_nominal.
%  Each perturbation source contributes a secular (linear in time) 
%  rate to one or more components of eps. The rates are summed at the
%  end to get the total secular ROE drift.
%
%  Convention: roe_rates = [deps_da; deps_dlam; deps_dex; deps_dey;
%                           deps_dix; deps_diy]  in [1/s]
% =========================================================================

% Initialize rate vectors for each perturbation source
roe_rates_J2   = zeros(6, 1);
roe_rates_SRP  = zeros(6, 1);
roe_rates_luni = zeros(6, 1);

% --- J2 Perturbation ---------------------------------------------------
%
% At GEO with the combiner-on-ellipse formation:
%   - Nominal delta_a = 0, so terms proportional to delta_a vanish
%   - Nominal delta_e = [0; 0], so J2 eccentricity precession acts on 
%     a zero vector and produces no first-order error
%   - The nonzero nominal delta_i_x couples into a secular delta_lambda 
%     drift via differential nodal regression
%
% From the J2 STM (Koenig, Guffanti & D'Amico 2017), the secular 
% differential ROE rates are:
%
%   deps_dlam_J2 = eta_J2 * n * (4 - 5*sin(i)^2) * delta_ix_nominal  (E13b)
%
% where eta_J2 = (3/4) * J2 * (R_e/a)^2                               (E14)
%
% This term arises because the nominal delta_ix (which sets the 
% cross-track oscillation amplitude) implies a small inclination 
% difference between collector and combiner. That inclination difference
% produces different RAAN precession rates, which accumulates as an 
% along-track separation drift.
%
% Physical magnitude at GEO:
%   eta_J2 = (3/4) * 1.08263e-3 * (6378.137/42164.17)^2 = 1.854e-5
%   deps_dlam_J2 = 1.854e-5 * 7.292e-5 * (4 - 5*sin(7.4deg)^2) * dlam*cos(phi0)/tan(beta)
%   For dlam = 2.37e-6 (100 m), beta = 60 deg, phi0 = 90 deg:
%     deps_dlam ~ 10^{-15} per second, giving ~3e-8 over a day
%     Corresponding OPD rate ~ a * 3e-8 * 1e3 * cos(beta) ~ 0.7 mm/day
%   This confirms J2 is a very small effect at GEO.
%
% All other J2 secular rates are zero to first order for this formation.
% -------------------------------------------------------------------------

if perturbation_flags.J2
    
    eta_J2 = (3/4) * J2 * (R_e / a)^2;                    % Eq. (E14)
    
    % Secular argument of perigee precession rate [rad/s]    Eq. (E11)
    omega_dot_J2 = eta_J2 * n * (5 * cos(inc)^2 - 1);
    
    % Secular RAAN precession rate [rad/s]                   Eq. (E12)
    RAAN_dot_J2 = -2 * eta_J2 * n * cos(inc);
    
    % Secular delta_lambda rate due to differential RAAN     Eq. (E13b)
    % The nominal delta_ix is the 5th component of roe_nominal
    dix_nominal = roe_nominal(5);
    deps_dlam_J2 = eta_J2 * n * (4 - 5 * sin(inc)^2) * dix_nominal;
    
    % Secular delta_e rates: zero because nominal delta_e = 0  Eq. (E13c-d)
    deps_dex_J2 = 0;
    deps_dey_J2 = 0;
    
    % Secular delta_a rate: zero                               Eq. (E13a)
    deps_da_J2 = 0;
    
    % Secular delta_i rates: zero because nominal delta_a = 0  Eq. (E13e-f)
    deps_dix_J2 = 0;
    deps_diy_J2 = 0;
    
    roe_rates_J2 = [deps_da_J2; deps_dlam_J2; deps_dex_J2; deps_dey_J2; ...
                    deps_dix_J2; deps_diy_J2];
    
end

%% --- SRP Perturbation ---------------------------------------------------
%
% PLACEHOLDER: To be implemented next.
%
% The differential SRP model is identical to the Hansen/Ireland case,
% but the coupling into OPD is different because the nominal delta_e
% is nonzero. The SRP-driven secular ROE rates will add to the J2 rates.
%
% Expected dominant effect: secular drift in delta_e magnitude and direction,
% scaling as (P_srp * delta_Bs) / (n * a).
% --------------------------------------------------------------------------

if perturbation_flags.SRP
    warning('compute_opd_analytical:notImplemented', ...
        'SRP perturbation model not yet implemented. Returning zero rates.');
    % roe_rates_SRP remains all zeros
end

%% --- Lunisolar Perturbation ---------------------------------------------
%
% PLACEHOLDER: To be implemented after SRP.
%
% Expected effect: secular drift in delta_e from lunar/solar tidal gradients,
% magnitude ~10^-8 m/s^2 at GEO, comparable to SRP.
% --------------------------------------------------------------------------

if perturbation_flags.lunisolar
    warning('compute_opd_analytical:notImplemented', ...
        'Lunisolar perturbation model not yet implemented. Returning zero rates.');
    % roe_rates_luni remains all zeros
end

%% ========================================================================
%  STEP 5: Sum all secular ROE rates
%  ========================================================================

roe_rates = roe_rates_J2 + roe_rates_SRP + roe_rates_luni;

fprintf('\n=== Total Secular ROE Rates ===\n');
fprintf('deps_da      = %.4e /s\n', roe_rates(1));
fprintf('deps_dlambda = %.4e /s\n', roe_rates(2));
fprintf('deps_dex     = %.4e /s\n', roe_rates(3));
fprintf('deps_dey     = %.4e /s\n', roe_rates(4));
fprintf('deps_dix     = %.4e /s\n', roe_rates(5));
fprintf('deps_diy     = %.4e /s\n', roe_rates(6));

%% ========================================================================
%  STEP 6: Compute OPD harmonic growth rates
%
%  From Rizza et al. (2026) Eqs. 7-8, the OPD as a function of the
%  combiner's mean argument of latitude u_c is:
%
%    OPD/a = A*cos^2(u) + B*sin^2(u) + C*sin(u)*cos(u) + D*cos(u) + E*sin(u)
%
%  where A, B, C, D, E are linear functions of the ROE. Rewriting using
%  double-angle identities:
%
%    OPD/a = (A+B)/2 + [D*cos(u) + E*sin(u)] + [(A-B)/2*cos(2u) + C/2*sin(2u)]
%          = OPD_DC   + OPD_1n                + OPD_2n
%
%  The ROE errors grow linearly: epsilon_ROE(t) = roe_rates * t.
%  Therefore each harmonic amplitude grows linearly, and the peak OPD
%  envelope is bounded by the sum of the amplitudes:
%
%    OPD_peak(t) ≤ |OPD_DC(t)| + OPD_1n_amp(t) + OPD_2n_amp(t)
%
%  Each amplitude has a growth rate (derivative with respect to time).
%  ========================================================================

% Unpack ROE error rates
deps_da   = roe_rates(1);
deps_dlam = roe_rates(2);
deps_dex  = roe_rates(3);
deps_dey  = roe_rates(4);
deps_dix  = roe_rates(5);
deps_diy  = roe_rates(6);

% A, B, C, D, E coefficients as functions of ROE errors (Rizza Eq. 8)
% Evaluated at the error rates (slopes of the coefficients vs. time)
A_eps_rate = -deps_dex * cos_beta * cos_phi0 - 2*deps_dey * cos_beta * sin_phi0;
B_eps_rate = -2*deps_dex * cos_beta * cos_phi0 - deps_dey * cos_beta * sin_phi0;
C_eps_rate = deps_dex * cos_beta * sin_phi0 + deps_dey * cos_beta * cos_phi0;
D_eps_rate = deps_da * cos_beta * cos_phi0 + deps_dlam * cos_beta * sin_phi0 ...
             - deps_diy * sin_beta;
E_eps_rate = deps_da * cos_beta * sin_phi0 - deps_dlam * cos_beta * cos_phi0 ...
             + deps_dix * sin_beta;

% OPD harmonic growth rates (slopes of OPD vs. time) in [dimensionless/s]
opd_dc_rate_dimless   = (A_eps_rate + B_eps_rate) / 2;
opd_1n_rate_dimless   = sqrt(D_eps_rate^2 + E_eps_rate^2);
opd_2n_rate_dimless   = 0.5 * sqrt((A_eps_rate - B_eps_rate)^2 + C_eps_rate^2);

% Convert to [m/s]: multiply by semimajor axis [km] and 1000 [m/km]
opd_rate_dc  = abs(opd_dc_rate_dimless) * a * 1e3;     % [m/s]
opd_rate_1n  = opd_1n_rate_dimless * a * 1e3;          % [m/s]
opd_rate_2n  = opd_2n_rate_dimless * a * 1e3;          % [m/s]

% Total peak OPD growth rate: sum of all harmonic rates
opd_rate_total = opd_rate_dc + opd_rate_1n + opd_rate_2n;  % [m/s]

fprintf('\n=== OPD Growth Rates ===\n');
fprintf('OPD_DC rate  = %.4e m/s = %.4f mm/day\n', ...
    opd_rate_dc, opd_rate_dc * 86400 * 1e3);
fprintf('OPD_1n rate  = %.4e m/s = %.4f mm/day\n', ...
    opd_rate_1n, opd_rate_1n * 86400 * 1e3);
fprintf('OPD_2n rate  = %.4e m/s = %.4f mm/day\n', ...
    opd_rate_2n, opd_rate_2n * 86400 * 1e3);
fprintf('OPD_total rate = %.4e m/s = %.4f mm/day\n', ...
    opd_rate_total, opd_rate_total * 86400 * 1e3);

%% ========================================================================
%  STEP 7: Compute passive integration time (total OPD vs. delay line stroke)
%  ========================================================================

if opd_rate_total < eps
    % No perturbation growth; integration time limited by geometric OPD only
    if opd_geometric_peak > D_max
        T_passive = 0;  % Cannot observe this star at all
        warning('compute_opd_analytical:geometricOPDtooLarge', ...
            'Geometric OPD exceeds delay line stroke. T_passive = 0.');
    else
        T_passive = Inf;  % Infinite integration time
        warning('compute_opd_analytical:zeroGrowthRate', ...
            'No perturbation-driven OPD growth. T_passive = Inf.');
    end
else
    % Time until total OPD (geometric + perturbation) reaches delay line stroke
    % The perturbation envelope starts at opd_geometric_peak and grows linearly
    available_stroke = D_max - opd_geometric_peak;  % [m]
    
    if available_stroke <= 0
        T_passive = 0;  % Geometric OPD already exceeds stroke
        warning('compute_opd_analytical:noAvailableStroke', ...
            'Geometric OPD (%.2f m) already exceeds or equals delay line stroke (%.2f m).\n', ...
            opd_geometric_peak, D_max);
    else
        T_passive = available_stroke / opd_rate_total;  % [s]
    end
end

fprintf('\n=== Passive Integration Time ===\n');
fprintf('Delay line stroke D_max = %.2f m\n', D_max);
fprintf('Geometric baseline OPD (peak) = %.2f m\n', opd_geometric_peak);
fprintf('Available stroke for perturbation drift = %.2f m\n', D_max - opd_geometric_peak);
if isfinite(T_passive) && T_passive > 0
    fprintf('T_passive = %.2f s = %.2f hours = %.2f days\n', ...
        T_passive, T_passive / 3600, T_passive / 86400);
elseif T_passive == 0
    fprintf('T_passive = 0 (star not observable with current formation)\n');
else
    fprintf('T_passive = Inf (no perturbation-driven OPD growth)\n');
end


%% ========================================================================
%  STEP 8: Compute Total OPD envelope time history (geometric + perturbation)
%  ========================================================================

% Perturbation-driven OPD (secular growth from ROE errors)
opd_dc_hist  = abs(opd_dc_rate_dimless) * t_eval * a * 1e3;  % [m]
opd_1n_hist  = opd_1n_rate_dimless * t_eval * a * 1e3;       % [m]
opd_2n_hist  = opd_2n_rate_dimless * t_eval * a * 1e3;       % [m]

% Perturbation envelope (worst-case sum of harmonics)
opd_perturbation_envelope = opd_dc_hist + opd_1n_hist + opd_2n_hist;  % [m]

% Total OPD: geometric baseline + perturbation growth
% The geometric OPD oscillates; the perturbation envelope grows.
% Worst case is when they add constructively:
opd_envelope_total = opd_geometric_peak + opd_perturbation_envelope;  % [m]

% Also compute the instantaneous total OPD (geometric oscillation + perturbation)
opd_instantaneous = opd_geometric_hist + opd_perturbation_envelope;   % [m]

%% ========================================================================
%  STEP 9: Package outputs
%  ========================================================================

results = struct();
results.T_passive      = T_passive;
results.opd_rate_dc    = opd_rate_dc;
results.opd_rate_1n    = opd_rate_1n;
results.opd_rate_2n    = opd_rate_2n;
results.opd_rate_total = opd_rate_total;
results.roe_rates      = roe_rates;
results.roe_rates_J2   = roe_rates_J2;
results.roe_rates_SRP  = roe_rates_SRP;
results.roe_rates_luni = roe_rates_luni;
results.opd_dc_hist    = opd_dc_hist;
results.opd_1n_hist    = opd_1n_hist;
results.opd_2n_hist    = opd_2n_hist;
results.beta_star      = beta_star;
results.phi0_star      = phi0_star;
results.roe_nominal    = roe_nominal;
results.s_hat_eci      = s_hat_eci;
results.formation_type = 'linear';
results.opd_geometric_peak = opd_geometric_peak;
results.opd_geometric_pp   = opd_geometric_pp;
results.opd_geometric_mean = opd_geometric_mean;
results.opd_geometric_rms  = opd_geometric_rms;
results.opd_geometric_hist = opd_geometric_hist;
results.opd_perturbation_envelope = opd_perturbation_envelope;
results.opd_envelope_total = opd_envelope_total;
results.opd_instantaneous  = opd_instantaneous;

end