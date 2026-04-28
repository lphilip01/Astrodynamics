function [sun_unit, moon_unit, earth_unit, earth_ang_radius, in_eclipse] = ...
    compute_exclusion_zones(time, r_sc_eci, params)

    jd = juliandate(time);


    % planetEphemeris outputs km (keep everything in km)
    sun_pos  = params.ephem.rSun.';
    moon_pos = params.ephem.rMoon.';

    sun_vec  = sun_pos  - r_sc_eci;
    moon_vec = moon_pos - r_sc_eci;
    earth_vec = -r_sc_eci;

    sun_unit   = sun_vec  / norm(sun_vec);
    moon_unit  = moon_vec / norm(moon_vec);
    earth_range = norm(r_sc_eci);
    earth_unit = earth_vec / earth_range;

    if isfield(params,'const') && isfield(params.const,'R_earth_km')
        Re = params.const.R_earth_km;
    else
        Re = 6378.137;
    end

    earth_ang_radius = asind(min(1, Re/earth_range));

    % Simple cylindrical eclipse test (first-order)
    cosang = dot(sun_vec, earth_vec) / (norm(sun_vec)*norm(earth_vec));
    cosang = max(-1,min(1,cosang));
    sun_earth_sep = acosd(cosang);

    in_eclipse = (dot(sun_vec, earth_vec) > 0) & (sun_earth_sep < earth_ang_radius);
end