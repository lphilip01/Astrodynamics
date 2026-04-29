function roe_collectors = initCombinerOnEllipseFormation(a_chief_km, Bmax_km, theta_deg)
% INITCOMBINERONELLIPSEFORMATION  Quasi-nonsingular ROE for the
%   Pogorelyuk/Black "combiner-on-ellipse" interferometer formation.
%
%   Implements the formation from:
%     Pogorelyuk et al., "Laser Guided Space Interferometer,"
%       Proc. SPIE 12183, 2022, Eqs. (4)-(5).
%     Black, "Space Optical Interferometry..." PhD thesis, MIT, 2025,
%       Eq. (4.1).
%
%   The combiner spacecraft is the ROE chief.  It traces a 2:1 radial /
%   along-track ellipse in the CW frame (its absolute orbit has nonzero
%   eccentricity ~ Bmax/(a*sqrt(12))).  Each collector has a circular
%   absolute orbit with a fixed along-track offset and cross-track
%   oscillation.  The optical path length from an in-plane target star
%   through every collector to the combiner is identically Bmax/sqrt(3).
%
%   ROE convention (D'Amico 2010 / Koenig et al. 2017, quasi-nonsingular):
%     da      = (a_d - a_c) / a_c
%     dlambda = (u_d - u_c) + (Om_d - Om_c)*cos(i_c)
%     dex     = ex_d - ex_c          (relative eccentricity, x)
%     dey     = ey_d - ey_c          (relative eccentricity, y)
%     dix     = i_d  - i_c           (relative inclination,  x)
%     diy     = (Om_d - Om_c)*sin(i_c)  (relative inclination, y)
%
%   INPUTS
%     a_chief_km  – combiner semi-major axis [km]  (e.g. 42164 for GEO)
%     Bmax_km     – maximum interferometric baseline [km]
%     theta_deg   – (N-1) x 1 vector of collector phase angles [deg]
%                   For 2 collectors: e.g. [0; 120]
%                   For 3 collectors: e.g. [0; 120; 240]
%
%   OUTPUT
%     roe_collectors – 6 x (N-1) matrix of dimensionless ROE
%                      [da; dlambda; dex; dey; dix; diy] per collector
%
%   UNITS  km in, dimensionless ROE out.
%
%   PASSIVE SAFETY NOTE
%     delta_e is along the x-axis; delta_i is along the y-axis for
%     each collector, so delta_e ⊥ delta_i.  The e/i vector separation
%     minimum RN-plane distance is therefore ZERO.  Collision avoidance
%     relies on the along-track offset a*|dlambda|.

% --- input processing ------------------------------------------------
Ncoll = numel(theta_deg);
theta_rad = deg2rad(theta_deg(:));   % column vector, rad

% --- ROE for each collector relative to combiner chief ----------------
%   da      = 0
%   dlambda = Bmax/(2*a) * cos(theta_i)
%   dex     = Bmax/(a*sqrt(12))            <-- nonzero: key difference
%   dey     = 0
%   dix     = 0
%   diy     = -Bmax/(2*a) * sin(theta_i)

da      = zeros(1, Ncoll);
dlambda = (Bmax_km / (2 * a_chief_km)) .* cos(theta_rad)';  % row vector
dex     = (Bmax_km / (a_chief_km * sqrt(12))) * ones(1, Ncoll);
dey     = zeros(1, Ncoll);
dix     = zeros(1, Ncoll);
diy     = -(Bmax_km / (2 * a_chief_km)) .* sin(theta_rad)';  % row vector

% --- Assemble output matrix -------------------------------------------
roe_collectors = [da; dlambda; dex; dey; dix; diy];  % 6 x Ncoll

end