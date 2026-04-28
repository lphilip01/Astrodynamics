%% sky_coverage_runner.m
clear; close all; clc;

%% Setup
params = setup_mission_parameters();

t_start = datetime(2026,1,1,0,0,0);
t_end   = t_start + days(365);
dt_analysis = 3600; % [s]

time_analysis = (t_start:seconds(dt_analysis):t_end).';
n_times = numel(time_analysis);
t_sec = seconds(time_analysis - time_analysis(1));

% RA/Dec grid (regular grid so existing metrics/figures work directly)
ra_grid  = 0:2:358;
dec_grid = -90:2:90;
[RA_mesh, DEC_mesh] = meshgrid(ra_grid, dec_grid);

ra_rad  = deg2rad(RA_mesh(:));
dec_rad = deg2rad(DEC_mesh(:));
target_unit = [cos(dec_rad).*cos(ra_rad), ...
               cos(dec_rad).*sin(ra_rad), ...
               sin(dec_rad)];
N_targets = size(target_unit,1);

%% Chief perturbation propagation params (km, s, rad)
paramsChief.mu      = params.const.mu_earth_km;
paramsChief.RE      = params.const.R_earth_km;
paramsChief.J2      = params.const.J2;
paramsChief.muMoon  = params.const.mu_moon_km;
paramsChief.muSun   = params.const.mu_sun_km;  % corrected value
paramsChief.CR      = params.dyn.CR;
paramsChief.As      = params.dyn.As;
paramsChief.m       = params.dyn.m;
paramsChief.S       = params.dyn.S;
paramsChief.c       = params.dyn.c;
paramsChief.jd0     = juliandate(t_start);
paramsChief.ephemModel = params.dyn.ephemModel;
paramsChief.useShadow  = params.dyn.useShadow;

% Initial chief QNS state [a; ex; ey; i; RAAN; u]
i0 = params.chief.i;
x0 = [params.chief.a_km; params.chief.ex; params.chief.ey; ...
      i0; params.chief.RAAN; params.chief.u];

% Precompute ephem for propagation
paramsChief.ephem = precompute_ephemeris(t_sec, paramsChief);
params.ephem=paramsChief.ephem;

T0 = 2*pi*sqrt(params.chief.a_km^3 / paramsChief.mu);
opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

fprintf('Propagating chief with perturbations...\n');
[~, x_qns] = ode45(@(t,x) rates_qns_total(t,x,paramsChief), t_sec, x0, opts);

% Convert QNS history -> ECI history
sc_positions_ECI = zeros(n_times,3);
sc_velocities_ECI = zeros(n_times,3);
for k = 1:n_times
    [r_eci, v_eci] = qns_to_eci(x_qns(k,:).', paramsChief.mu);
    sc_positions_ECI(k,:) = r_eci.';
    sc_velocities_ECI(k,:) = v_eci.';
end

%% Coverage computation (vectorized over targets)
fprintf('Computing sky visibility...\n');

r_sc_eci = sc_positions_ECI(:,:).';
[sun_unit, moon_unit, earth_unit, earth_ang_radius, in_eclipse] = ...
    compute_exclusion_zones(time_analysis, r_sc_eci, params);

% vectorized over all sky targets
visibility_history = check_target_visibility( ...
    target_unit, r_sc_eci, sun_unit, moon_unit, earth_unit, ...
    earth_ang_radius, in_eclipse, params);

%% Per-target statistics
coverage_count_vec = sum(visibility_history,2);
total_duration_vec = coverage_count_vec * dt_analysis;

max_continuous_vec = zeros(N_targets,1);
revisit_times_vec  = cell(N_targets,1);

for idx = 1:N_targets
    seq = visibility_history(idx,:).';
    [max_continuous_vec(idx), ~] = find_max_continuous(seq, dt_analysis);

    d = diff([false; seq; false]);
    starts = find(d==1);
    ends   = find(d==-1)-1;
    if numel(starts) >= 2
        gaps = (starts(2:end) - ends(1:end-1) - 1) * dt_analysis;
        revisit_times_vec{idx} = gaps(gaps>0);
    else
        revisit_times_vec{idx} = [];
    end
end

%% Reshape to 2D maps for existing metrics/figures functions
sz = size(RA_mesh);
coverage_count = reshape(coverage_count_vec, sz);
total_duration = reshape(total_duration_vec, sz);
max_continuous = reshape(max_continuous_vec, sz);
revisit_times  = reshape(revisit_times_vec,  sz);

%% Metrics + figures (existing pipeline)
metrics = compute_coverage_metrics(coverage_count, total_duration, ...
    max_continuous, revisit_times, n_times, dt_analysis, RA_mesh, DEC_mesh);

fprintf('\n=== Sky Coverage Summary ===\n');
fprintf('Total sky coverage: %.2f%%\n', metrics.total_coverage_percent);
fprintf('Mean observation efficiency: %.2f%%\n', metrics.mean_efficiency);
fprintf('Ecliptic coverage: %.2f%%\n', metrics.ecliptic_coverage);
fprintf('Galactic coverage: %.2f%%\n', metrics.galactic_coverage);
fprintf('Median max continuous: %.2f h\n', metrics.median_max_cont_hours);
fprintf('Median revisit: %.2f h\n', metrics.median_revisit_hours);

generate_figures(RA_mesh, DEC_mesh, coverage_count, total_duration, ...
    max_continuous, revisit_times, metrics, params, n_times);

if params.save
save('sky_coverage_results.mat', ...
    'metrics','coverage_count','total_duration','max_continuous','revisit_times', ...
    'RA_mesh','DEC_mesh','visibility_history','x_qns', ...
    'sc_positions_ECI','sc_velocities_ECI','time_analysis','params','paramsChief','-v7.3');
end
fprintf('\nDone.\n');