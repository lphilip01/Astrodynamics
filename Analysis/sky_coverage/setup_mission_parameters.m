function params = setup_mission_parameters(inc,raan)

% Exclusions [deg]
params.exclusion.earth_margin = 15;
params.exclusion.sun  = 60;
params.exclusion.moon = 30;

% Constants (km-based)
params.const.mu_earth_km = 398600.4418;
params.const.R_earth_km  = 6378.137;
params.const.J2          = 1082.63e-6;
params.const.mu_moon_km  = 4902.800066;
params.const.mu_sun_km   = 1.32712440018e11;

% Chief initial QNS
params.chief.a_km = 42164;
params.chief.ex   = 1e-4;
params.chief.ey   = 0;
params.chief.i    = deg2rad(inc);
params.chief.RAAN = deg2rad(raan);
params.chief.u    = 0;

% Dynamics settings
params.dyn.CR = 2;
params.dyn.As = 2;         % m^2
params.dyn.m  = 200;       % kg
params.dyn.S  = 1367;      % W/m^2
params.dyn.c  = 2.998e8;   % m/s
params.dyn.ephemModel = '421';
params.dyn.useShadow  = true;

% Viz
params.viz.save_figs = true;
params.viz.fig_dir = strcat('figures_',num2str(inc),'_',num2str(raan));
if ~exist(params.viz.fig_dir,'dir') 
    mkdir(params.viz.fig_dir); 
end
params.save=true;

end