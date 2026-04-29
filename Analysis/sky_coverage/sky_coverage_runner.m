function metrics=sky_coverage_runner(inc,raan)
%% sky_coverage_runner.m

%% Setup
params = setup_mission_parameters(inc,raan);

t_start = datetime(2026,1,1,0,0,0);
t_end   = t_start + days(365);
dt_analysis = 3600; % [s]

time_analysis = (t_start:seconds(dt_analysis):t_end).';
n_times = numel(time_analysis);
t_sec = seconds(time_analysis - time_analysis(1));

% Equal-area sky sampling via Fibonacci sphere
N_targets = 16290; % ~equivalent point density to the 2-deg regular grid (180x91)
[RA_deg, DEC_deg, target_unit] = generate_equal_area_sky(N_targets);

% Column vectors used as coordinate labels (replaces RA_mesh / DEC_mesh)
RA_mesh  = RA_deg(:);
DEC_mesh = DEC_deg(:);

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
params.ephem.rSun = paramsChief.ephem.rSun;  
params.ephem.rMoon = paramsChief.ephem.rMoon;

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
visibility_history = false(N_targets, n_times);

for k = 1:n_times
    if mod(k,200)==0
        fprintf('  step %d/%d (%.1f%%)\n', k, n_times, 100*k/n_times);
    end

    r_sc_eci = sc_positions_ECI(k,:).';
    [sun_unit, moon_unit, earth_unit, earth_ang_radius, in_eclipse] = ...
        compute_exclusion_zones(time_analysis(k), r_sc_eci, params,k);

    % vectorized over all sky targets
    visibility_history(:,k) = check_target_visibility( ...
        target_unit, r_sc_eci, sun_unit, moon_unit, earth_unit, ...
        earth_ang_radius, in_eclipse, params);
end

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

%% Pass flat vectors directly (no reshape needed for equal-area sampling)
coverage_count = coverage_count_vec;
total_duration = total_duration_vec;
max_continuous = max_continuous_vec;
revisit_times  = revisit_times_vec;

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

save(strcat('sky_coverage_results_',num2str(inc),'_',num2str(raan),'.mat'), ...
    'metrics','coverage_count','total_duration','max_continuous','revisit_times', ...
    'RA_mesh','DEC_mesh','visibility_history','x_qns', ...
    'sc_positions_ECI','sc_velocities_ECI','time_analysis','params','paramsChief','-v7.3');
fprintf('\nDone.\n');
end