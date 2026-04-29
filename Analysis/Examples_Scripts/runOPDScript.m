%% Parameter sweep over all combinations
altair_ra=5.1832;
altair_dec=0.1548;

deneb_ra=5.403;
deneb_dec=0.800;

% Fixed star direction
tau_ceti_ra  = 0.4600;
tau_ceti_dec = -0.2781;

% Fixed other params
T    = 0.02;
dmax = 5;

% Sweep vectors
rho_m_vals  = [500, 1000, 2500];          % m
Tint_s_vals = [30*60, 60*60, 180*60];     % s

% Area-to-mass ratios: fix mass, vary area
am_ratios = [0.003, 0.007, 0.015];        % m^2/kg

mc = 200;  % kg, chief mass (fixed)
md = 200;  % kg, deputy mass (fixed)

Asc_vals = am_ratios * mc;   % [0.6, 1.4, 3.0] m^2
Asd_vals = am_ratios * md;   % [0.6, 1.4, 3.0] m^2

% Create results folder
results_dir = 'results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Count total runs
n_total = numel(rho_m_vals) * numel(Tint_s_vals) * numel(Asc_vals) * numel(Asd_vals);
run_idx = 0;

%% Main loop — full grid (3x3x3x3 = 81 runs)
for i_rho = 1:numel(rho_m_vals)
    rho_m = rho_m_vals(i_rho);

    for i_tint = 1:numel(Tint_s_vals)
        Tint_s = Tint_s_vals(i_tint);

        for i_asc = 1:numel(Asc_vals)
            Asc = Asc_vals(i_asc);

            for i_asd = 1:numel(Asd_vals)
                Asd = Asd_vals(i_asd);

                run_idx = run_idx + 1;
                fprintf('[%d/%d] rho=%dm  Tint=%dmin  Asc=%.2f  Asd=%.2f\n', ...
                    run_idx, n_total, rho_m, Tint_s/60, Asc, Asd);

                % Run simulation
                [out, sol] = Example_GEO_Formation_OPD( ...
                    tau_ceti_ra, tau_ceti_dec, ...
                    rho_m, Tint_s, Asc, mc, Asd, md, T, dmax);

                % Build filename tag
tag = sprintf('rho%d_Tint%d_Asc%d_mc%d_Asd%d_md%d', ...
    rho_m, round(Tint_s/60), round(Asc*1000), mc, round(Asd*1000), md);

                % Save to results folder
                save(fullfile(results_dir, ['out_' tag]), 'out');
                save(fullfile(results_dir, ['sol_' tag]), 'sol');
            end
        end
    end
end

fprintf('Done. %d runs saved to ''%s/''.\n', run_idx, results_dir);