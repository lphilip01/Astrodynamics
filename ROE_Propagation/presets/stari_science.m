function [chief, roe0, star] = stari_science()
%STARI_SCIENCE  STARI science orbit ROE configuration.
%
% Implements the science configuration from the STARI paper (Rizza et al. 2026).
% The formation is designed so the interferometric baseline sweeps perpendicular
% to the target star direction over one orbit.
%
% Key design principle (STARI perpendicularity condition):
%   For a star at azimuth phi_0 and elevation beta, the inclination vector
%   should satisfy: diy/dix = tan(phi_0), with |di| set by baseline requirement.
%   The T-N oscillation is then perpendicular to the star's projection in T-N plane.
%
% This preset: phi_0 = 90 deg (star in T-N plane, along-track direction),
%              beta = 45 deg (elevation, gives diy = dlambda * sin(90)/tan(45) = dlambda)
%
% Outputs:
%   chief - GEO chief orbital elements struct
%   roe0  - Initial absolute ROE [m] (science orbit)
%   star  - Star pointing parameters struct

    % GEO chief (same as geo_default)
    chief.a     = 42164e3;
    chief.e     = 0.0;
    chief.i     = 0.0;
    chief.Omega = 0.0;
    chief.omega = 0.0;
    chief.M0    = 0.0;
    chief.mu    = 3.986004418e14;

    % STARI science ROE: pure T-N oscillation perpendicular to star
    % dlambda provides baseline scale, diy provides cross-track component
    roe0.da      = 0;     % [m] drift-free
    roe0.dlambda = 100;   % [m] along-track mean separation (sets T baseline)
    roe0.dex     = 0;     % [m]
    roe0.dey     = 0;     % [m]
    roe0.dix     = 0;     % [m] no R-N inclination x (phi_0=90 -> dix=0)
    roe0.diy     = 100;   % [m] cross-track amplitude = dlambda for beta=45

    % Star pointing parameters
    star.phi_0 = 90;    % [deg] azimuth from R toward T (star along T-axis)
    star.beta  = 45;    % [deg] elevation above R-T plane
end
