altair_ra=5.1832;
altair_dec=0.1548;

deneb_ra=5.403;
deneb_dec=0.800;

% Fixed star direction
tau_ceti_ra  = 0.4600;
tau_ceti_dec = -0.2781;

% Fixed other params
T    = 0.02; %thrust in Newtons
dmax = 5; %max optical delay line

% Sweep vectors
rho_m  = 500;          % m
Tint_s = 60*60;     % s

% Area-to-mass ratios: fix mass, vary area
am_ratio_c = 0.007;        % m^2/kg
am_ratio_d = 0.003;

mc = 200;  % kg, chief mass (fixed)
md = 200;  % kg, deputy mass (fixed)

Asc = am_ratio_c * mc;   % [0.6, 1.4, 3.0] m^2
Asd = am_ratio_d * md;   % [0.6, 1.4, 3.0] m^2

[out, sol]=Example_GEO_Formation_OPD(tau_ceti_ra,tau_ceti_dec,rho_m,Tint_s,Asc,mc,Asd,md,T,dmax)


