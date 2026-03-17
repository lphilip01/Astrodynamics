function [chief, roe0] = leo_sso()
%LEO_SSO  Sun-synchronous LEO chief orbit preset.
%
% Chief orbit: 500 km altitude, sun-synchronous inclination (~97.4 deg).
% Representative of PRISMA/PROBA-3 type formation flying missions.
% Orbital period: ~94.7 minutes (much shorter than GEO ~24 hours).
%
% The LEO environment introduces significant J2 perturbations (RAAN precession,
% argument of perigee drift) not captured in the Keplerian model here.
% Use this preset to compare ROE dynamics at LEO vs GEO time scales.
%
% Outputs:
%   chief - Chief orbital elements struct
%   roe0  - Initial absolute ROE struct [m]

    chief.a     = 6878e3;            % semi-major axis [m] = R_E + 500 km
    chief.e     = 0.0;               % eccentricity (circular)
    chief.i     = 97.4;              % inclination [deg], SSO condition
    chief.Omega = 0.0;               % RAAN [deg]
    chief.omega = 0.0;               % argument of perigee [deg]
    chief.M0    = 0.0;               % initial mean anomaly [deg]
    chief.mu    = 3.986004418e14;    % Earth gravitational parameter [m^3/s^2]

    % Initial ROE: STARI-like configuration scaled to LEO
    roe0.da      = 0;     % [m] drift-free
    roe0.dlambda = 500;   % [m] along-track offset
    roe0.dex     = 0;     % [m]
    roe0.dey     = 0;     % [m]
    roe0.dix     = 0;     % [m]
    roe0.diy     = 100;   % [m] cross-track amplitude
end
