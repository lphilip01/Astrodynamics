function [chief, roe0] = geo_default()
%GEO_DEFAULT  Default chief orbit and ROE for GEO interferometry scenario.
%
% Chief orbit: Geostationary, circular, equatorial.
% Initial ROE: along-track separation + cross-track inclination (STARI-like).
%
% Outputs:
%   chief - Chief orbital elements struct
%   roe0  - Initial absolute ROE struct [m]

    chief.a     = 42164e3;           % semi-major axis [m], GEO radius
    chief.e     = 0.0;               % eccentricity (circular)
    chief.i     = 0.0;               % inclination [deg]
    chief.Omega = 0.0;               % RAAN [deg]
    chief.omega = 0.0;               % argument of perigee [deg]
    chief.M0    = 0.0;               % initial mean anomaly [deg]
    chief.mu    = 3.986004418e14;    % Earth gravitational parameter [m^3/s^2]

    % Default ROE: along-track separation + cross-track oscillation
    roe0.da      = 0;     % [m] no differential SMA (no drift)
    roe0.dlambda = 500;   % [m] 500 m along-track separation
    roe0.dex     = 0;     % [m] no relative eccentricity
    roe0.dey     = 0;     % [m]
    roe0.dix     = 0;     % [m] no relative inclination x
    roe0.diy     = 100;   % [m] 100 m cross-track oscillation amplitude
end
