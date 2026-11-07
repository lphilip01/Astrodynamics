function [phi0, beta, sRTN0] = radec_to_phibeta(ra, dec, RAAN, inc)
% Convert a fixed star direction from ECI RA/Dec to chief RTN coordinates
% at the ascending node (u = 0). All input and output angles are radians.
%
% Right-handed RTN convention: Nhat = Rhat x That.
% For fixed inc and RAAN:
%   sRTN(u) = [cos(beta)*cos(phi0-u);
%              cos(beta)*sin(phi0-u);
%              sin(beta)];

sI = [cos(dec)*cos(ra);
      cos(dec)*sin(ra);
      sin(dec)];

Rhat0 = [ cos(RAAN);
          sin(RAAN);
          0 ];

That0 = [-sin(RAAN)*cos(inc);
          cos(RAAN)*cos(inc);
          sin(inc) ];

Nhat0 = [ sin(RAAN)*sin(inc);
         -cos(RAAN)*sin(inc);
          cos(inc) ];

sRTN0 = [dot(sI, Rhat0);
         dot(sI, That0);
         dot(sI, Nhat0)];

phi0 = atan2(sRTN0(2), sRTN0(1));
beta = asin(max(-1, min(1, sRTN0(3))));
end