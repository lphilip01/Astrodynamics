function [phi0, beta, sRTN0] = radec_to_phibeta(ra, dec, RAAN, inc)
% radec_to_phibeta
%
% Convert star right ascension / declination into (phi0, beta)
% used by the RTN-based interferometer geometry.
%
% Inputs:
%   ra   - star right ascension [rad]
%   dec  - star declination [rad]
%   RAAN - chief orbit RAAN [rad]
%   inc  - chief orbit inclination [rad]
%
% Outputs:
%   phi0  - star azimuth at chief ascending node [rad]
%   beta  - star elevation above chief orbital plane [rad]
%   sRTN0 - star unit vector in RTN at ascending node [3x1]
%
% Definitions:
%   sRTN(u) = [cos(beta)*cos(phi0-u);
%              cos(beta)*sin(phi0-u);
%              sin(beta)]
%
% So phi0 is the RTN azimuth when chief argument of latitude u = 0.

% Star inertial unit vector
sI = [cos(dec)*cos(ra);
      cos(dec)*sin(ra);
      sin(dec)];

% RTN basis at chief ascending node (u = 0)
Rhat0 = [cos(RAAN);
         sin(RAAN);
         0];

That0 = [-sin(RAAN)*cos(inc);
          cos(RAAN)*cos(inc);
          sin(inc)];

Nhat0 = [ -sin(RAAN)*sin(inc);
         cos(RAAN)*sin(inc);
          -cos(inc)];

% Star in RTN at ascending node
sR = dot(sI, Rhat0);
sT = dot(sI, That0);
sN = dot(sI, Nhat0);

sRTN0 = [sR; sT; sN];

% Elevation and azimuth
beta = asin(max(-1,min(1,sN)));
phi0 = atan2(sT, sR);

end