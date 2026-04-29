function light_switch = los_qns(r_sat, r_sun, RE)
% los_qns
%
% Returns:
%   light_switch = 1 if illuminated
%                = 0 if in Earth's shadow
%

rsat = norm(r_sat);
rsun = norm(r_sun);

theta     = acos( dot(r_sat, r_sun)/(rsat*rsun) );
theta_sat = acos( RE/rsat );
theta_sun = acos( RE/rsun );

if (theta_sat + theta_sun) <= theta
    light_switch = 0;   % eclipse
else
    light_switch = 1;   % illuminated
end
end