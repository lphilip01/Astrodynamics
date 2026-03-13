function delta_alpha_dot = ROE_dynamics_J2_EP(delta_alpha, u_RTN, uc, auxdata)
% =========================================================================
% ROE_dynamics_J2_EP.m
%
% Time derivative of quasi-nonsingular Relative Orbital Elements (ROE)
% under J2 perturbation + continuous EP thrust.
%
% Compatible with numeric (double) and CasADi SX/MX symbolic inputs.
%
% State: delta_alpha = [da; dlambda; dex; dey; dix; diy]  (dimensionless,
%        i.e. normalized by semi-major axis a_c)
%
%   da      = (a_d - a_c) / a_c
%   dlambda = relative mean longitude
%   dex     = e_xd - e_xc
%   dey     = e_yd - e_yc
%   dix     = i_d  - i_c
%   diy     = (Omega_d - Omega_c) * sin(i_c)
%
% Inputs:
%   delta_alpha  [6x1]  ROE state (dimensionless)
%   u_RTN        [3x1]  EP specific acceleration [m/s^2] in RTN
%   uc           [1x1]  chief mean argument of latitude [rad]
%   auxdata      struct  orbital constants
%
% Output:
%   delta_alpha_dot  [6x1]  ROE rates [1/s]
%
% Dynamics = J2 secular drift + EP control contribution
%
% References:
%   D'Amico (2010) PhD Thesis, TU Delft
%   Koenig, Guffanti, D'Amico (2017) JGCD
%   Rizza et al. (2026) STARI paper, Eq.(22),(25)-(27)
% =========================================================================

n   = auxdata.n;       % chief mean motion [rad/s]
ac  = auxdata.ac;      % chief semi-major axis [m]
ic  = auxdata.ic;      % chief inclination [rad]
J2  = auxdata.J2;
Re  = auxdata.Re;

% Unpack ROE
da      = delta_alpha(1);
dlambda = delta_alpha(2);
dex     = delta_alpha(3);
dey     = delta_alpha(4);
% dix   = delta_alpha(5);  % not needed for drift computation
% diy   = delta_alpha(6);

% =========================================================================
% J2 secular drift terms
% =========================================================================

% J2 precession rate of argument of perigee + RAAN (chief orbit)
% This drives eccentricity vector rotation and RAAN drift
kJ2 = (3/4) * n * (Re/ac)^2 * J2;

% Rate of eccentricity vector phase rotation (Eq. 27 in STARI paper)
phi_dot_J2 = kJ2 * (5*cos(ic)^2 - 1);

% Along-track drift due to relative semi-major axis
% (dominant secular term: d(dlambda)/dt = -3/2 * n * da)
% Plus J2 correction term for differential J2 nodal regression
eta = sqrt(1 - 0);  % near-circular: eta ~ 1
gamma = kJ2 * cos(ic);   % RAAN precession rate contribution

% Full J2 mean ROE secular drift (linearized about chief):
% Reference: D'Amico (2010) Eq. 3.41, simplified for circular chief
dlambda_dot_J2  = -(3/2)*n*da + (7/2)*kJ2*sin(ic)^2 * ...
                   (dex*sin(uc) - dey*cos(uc));

% Eccentricity vector rotates at phi_dot_J2 (counter-clockwise in dex-dey plane)
dex_dot_J2 = -phi_dot_J2 * dey;
dey_dot_J2 =  phi_dot_J2 * dex;

% Inclination vector: dix secular drift is zero for delta_ix = 0 condition
% diy has a slow drift due to differential J2 RAAN precession which is
% second order for near-identical orbits — neglected here
dix_dot_J2 = 0;
diy_dot_J2 = 0;

% da has no secular drift under J2 (energy is conserved in mean sense)
da_dot_J2   = 0;

J2_drift = [da_dot_J2;
            dlambda_dot_J2;
            dex_dot_J2;
            dey_dot_J2;
            dix_dot_J2;
            diy_dot_J2];

% =========================================================================
% EP control contribution: d(delta_alpha)/dt = B_cont * u_RTN
%
% For continuous acceleration a_RTN [m/s^2], the instantaneous ROE rates
% are given by the same GVE-based B matrix as for impulsive maneuvers
% (Eq. 22 in STARI paper), now interpreted as rates rather than jumps.
%
% B_cont = (1/n/ac) * [0,      2,          0;
%                      -2,    -3*(uc-uk),   0;     <- uk=uc for continuous
%                       sin(uc), 2*cos(uc), 0;
%                      -cos(uc), 2*sin(uc), 0;
%                       0,       0,    cos(uc);
%                       0,       0,    sin(uc)]
%
% For continuous thrust uk = uc (maneuver happening right now),
% so the (uc - uk) term = 0.
% =========================================================================

B_cont = (1/(n*ac)) * ...
    [0,        2,          0;
    -2,        0,          0;
     sin(uc),  2*cos(uc),  0;
    -cos(uc),  2*sin(uc),  0;
     0,        0,          cos(uc);
     0,        0,          sin(uc)];

EP_rates = B_cont * u_RTN;

% =========================================================================
% Total ROE rate
% =========================================================================
delta_alpha_dot = J2_drift + EP_rates;

end
