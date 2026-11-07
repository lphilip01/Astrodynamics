regulus_ra=2.6545;
regulus_dec=0.20887;

alphecca_ra=4.0784;
alphecca_dec=0.4663;

altais_ra=5.02899;
altais_dec=1.18094;

% Fixed star direction
tau_ceti_ra  = 0.454;
tau_ceti_dec = -0.2781;

eltanin_ra    = 269.15*pi/180;      
eltanin_dec    = 51.49*pi/180;  % Gamma Draconis (Eltanin)

% Fixed other params
T    = 0.04; %thrust in Newtons
dmax = 5; %max optical delay line

% Sweep vectors
rho_m  = 1000;          % circumradius to deputies m
Tint_s = 120*60;     % integration time s
settle_time=3600;

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

 [out, sol]=Example_GEO_Formation_OPD(tau_ceti_ra,tau_ceti_dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,settle_time,plot_figs,optimize);
 
 %Sky_Coverage_Runner(inc,0);
 if sol.success
 save('sol_7200_int_3600_settle_tau_ceti','sol');
 save('out_7200_int_3600_settle_tau_ceti','out');
 end
 
 Tint_s = 120*60;     % integration time s
 [out, sol]=Example_GEO_Formation_OPD(tau_ceti_ra,tau_ceti_dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,settle_time,plot_figs,optimize);
 
 if sol.success
 save('sol_7200_int_1800_settle_tau_ceti','sol');
 save('out_7200_int_1800_settle_tau_ceti','out');
 end

%% Other stars

Tint_s = 120*60;     % integration time s
[out, sol]=Example_GEO_Formation_OPD(regulus_ra,regulus_dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,settle_time,plot_figs,optimize);

if sol.success
save('sol_7200_int_1800_settle_regulus','sol');
save('out_7200_int_1800_settle_regulus','out');
end


Tint_s = 120*60;     % integration time s
[out, sol]=Example_GEO_Formation_OPD(alphecca_ra,alphecca_dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,settle_time,plot_figs,optimize);

if sol.success
save('sol_7200_int_1800_settle_alphecca','sol');
save('out_7200_int_1800_settle_alphecca','out');
end

Tint_s = 120*60;     % integration time s
[out, sol]=Example_GEO_Formation_OPD(eltanin_ra,eltanin_dec,rho_m,Tint_s,inc,Asc,mc,Asd,md,T,dmax,settle_time,plot_figs,optimize);

if sol.success
save('sol_7200_int_1800_settle_eltanin','sol');
save('out_7200_int_1800_settle_eltanin','out');
end
% flight_safety(out)
% 
 opts = struct();
 opts.Dmax_m = dmax;
 opts.pauseTime = 0.02;
 opts.videoFile = 'test_new_formation.mp4';

%animate_full_solution_presentation(out, sol, opts);

