function ephem = precompute_ephemeris(t, params)
% precompute_ephemeris
%
% Vectorized precomputation of Sun and Moon ephemerides on time grid t.
%
% Inputs:
%   t      : Nx1 or 1xN time vector [s]
%   params : struct with
%            params.jd0
%            params.ephemModel (optional, default '421')
%
% Output:
%   ephem.t      : Nx1 time vector [s]
%   ephem.rSun   : Nx3 Sun position wrt Earth [km]
%   ephem.rMoon  : Nx3 Moon position wrt Earth [km]
%   ephem.method : interpolation method ('linear')
%

t = t(:);

if isfield(params,'ephemModel')
    ephModel = params.ephemModel;
else
    ephModel = '421';
end

% Julian dates corresponding to each time sample
jd = params.jd0 + t/86400;

% Vectorized calls to planetEphemeris
ephem.t     = t;
ephem.rSun  = planetEphemeris(jd, 'Earth', 'Sun',  ephModel);
ephem.rMoon = planetEphemeris(jd, 'Earth', 'Moon', ephModel);
ephem.method = 'linear';
end