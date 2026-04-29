altair_ra=5.1832;
altair_dec=0.1548;

deneb_ra=5.403;
deneb_dec=0.800;

% Fixed star direction
tau_ceti_ra  = 0.4600;
tau_ceti_dec = -0.2781;

% Fixed other params
T    = 0.04; %thrust in Newtons
dmax = 5; %max optical delay line

% Sweep vectors
rho_m  = 1000;          % circumradius to deputies m
Tint_s = 60*60;     % integration time s

inc=24; %inclination in degrees

% Area-to-mass ratios: fix mass, vary area
am_ratio_c = 0.003;        % m^2/kg
am_ratio_d = 0.005;

mc = 613;  % kg, chief mass (fixed)
md = 262;  % kg, deputy mass (fixed)

Asc = am_ratio_c * mc;   % 
Asd = am_ratio_d * md;   % 

plot_figs=0;
optimize=1;
[out, sol]=Example_GEO_Formation_OPD(tau_ceti_ra,tau_ceti_dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,plot_figs,optimize);

%flight_safety(out)

opts = struct();
opts.Dmax_m = dmax;
opts.pauseTime = 0.02;
opts.videoFile = 'science_presentation_deneb.mp4';

animate_full_solution_presentation(out, sol, opts);

