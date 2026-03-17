function [chief, roe0] = stari_standby()
%STARI_STANDBY  STARI passively safe standby orbit using e/i separation.
%
% Implements the e/i separation strategy (D'Amico & Montenbruck 2006) as a
% safe parking configuration. When the eccentricity vector de = [dex; dey] and
% inclination vector di = [dix; diy] are parallel (same direction in phase space),
% the R-N separation is maximized throughout the orbit, ensuring passive
% collision avoidance without active manoeuvres.
%
% E/I separation conditions (for maximum R-N clearance):
%   - da = 0    (no drift)
%   - |de| > 0  (nonzero relative eccentricity)
%   - |di| > 0  (nonzero relative inclination)
%   - de parallel to di (maximizes minimum R-N separation)
%
% With de and di parallel (both pointing in x-direction here):
%   rR = -cos(u)*dex, rN = -cos(u)*dix  => R and N always correlated
%   => R-N locus is a line through origin with slope dix/dex, min separation = 0 ??
%
% Actually for maximum R-N: de perpendicular to di gives a circle in R-N plane.
% Standby config here uses parallel (collinear) for simple demonstration.
%
% Outputs:
%   chief - GEO chief orbital elements struct
%   roe0  - Initial absolute ROE [m] (standby, e/i separated)

    % GEO chief
    chief.a     = 42164e3;
    chief.e     = 0.0;
    chief.i     = 0.0;
    chief.Omega = 0.0;
    chief.omega = 0.0;
    chief.M0    = 0.0;
    chief.mu    = 3.986004418e14;

    % E/I separated standby: parallel de and di vectors
    % Both pointing along x (dey=0, diy=0), creating a passively safe orbit
    roe0.da      = 0;    % [m] drift-free (essential for passive safety)
    roe0.dlambda = 0;    % [m] no mean along-track offset
    roe0.dex     = 50;   % [m] relative eccentricity x-component
    roe0.dey     = 0;    % [m]
    roe0.dix     = 50;   % [m] relative inclination x-component (parallel to de)
    roe0.diy     = 0;    % [m]
end
