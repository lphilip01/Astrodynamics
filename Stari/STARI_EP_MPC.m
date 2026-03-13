% =========================================================================
% STARI_EP_MPC.m
%
% STARI Formation Flying — Station Keeping via Continuous EP Thrust
%
% Uses quasi-nonsingular ROE state representation from STARI paper.
% Replaces impulsive ΔV control with continuous EP thrust optimized
% via Model Predictive Control (MPC) in CasADi.
%
% Science orbit: S1 from Table 3 — phi0=90 deg, beta=45 deg
%   a|delta_lambda| = 100 m,  a|delta_iy| = 100 m,  all others = 0
%
% MPC problem:
%   Minimize:    sum ||u_k||^2 * dt   (fuel proxy)
%   Subject to:  ROE dynamics (J2 + EP)
%                Thrust bound: ||u|| <= a_max
%                ROE stays near science orbit (soft tracking)
%
% Dependencies: ROE_dynamics_J2_EP.m, STARI_science_orbit.m, CasADi
% =========================================================================

clc; clear; close all;
% addpath('/path/to/casadi')   % <-- update to your CasADi installation
import casadi.*

fprintf('==============================================\n');
fprintf(' STARI EP Formation Keeping — CasADi MPC\n');
fprintf('==============================================\n\n');

% =========================================================================
% 1. MISSION PARAMETERS (STARI LEO SSO)
% =========================================================================
mu_E = 3.986004418e14;   % [m^3/s^2]
Re   = 6378137.0;        % [m]
J2   = 1.08263e-3;

ac   = 6878e3;           % chief SMA [m] (500 km altitude)
ic   = 97.402 * pi/180;  % inclination [rad] (SSO)
ec   = 0;                % circular

n    = sqrt(mu_E / ac^3);      % mean motion [rad/s]
T_orb = 2*pi / n;              % orbital period [s]

fprintf('Chief orbit:\n');
fprintf('  a = %.0f km,  i = %.3f deg,  n = %.6e rad/s\n', ...
        ac/1e3, ic*180/pi, n);
fprintf('  T = %.2f min\n\n', T_orb/60);

auxdata.n  = n;
auxdata.ac = ac;
auxdata.ic = ic;
auxdata.J2 = J2;
auxdata.Re = Re;

% =========================================================================
% 2. EP THRUSTER PARAMETERS (representative cold gas / EP system)
%    STARI uses cold gas (VISORS heritage), but we model as low-thrust EP
%    to demonstrate the continuous control concept.
% =========================================================================
T_max    = 1e-3;         % Max thrust per axis [N]  (cold gas ~1 mN)
m_sc     = 8.0;          % CubeSat mass [kg]  (6U ~8 kg)
a_max    = T_max / m_sc; % Max specific acceleration [m/s^2]

fprintf('EP parameters:\n');
fprintf('  T_max  = %.2e N per axis\n',  T_max);
fprintf('  m_sc   = %.1f kg\n',          m_sc);
fprintf('  a_max  = %.4e m/s^2\n\n',     a_max);

% =========================================================================
% 3. SCIENCE ORBIT TARGET (S1: 100m baseline)
% =========================================================================
baseline_m  = 100;                   % desired along-track separation [m]
phi0_deg    = 90;                    % frozen orbit (J2 stable)
beta_deg    = 45;                    % star elevation (satisfies REQ.1)

delta_lambda_target = baseline_m / ac;  % dimensionless

[delta_alpha_star, orbit_info] = STARI_science_orbit(delta_lambda_target, ...
                                                      phi0_deg, beta_deg, ac);

% =========================================================================
% 4. SIMULATION SETUP
% =========================================================================
N_orbits   = 20;                     % simulate 20 orbits
dt         = 30;                     % propagation time step [s]
N_total    = round(N_orbits * T_orb / dt);
t_vec      = (0:N_total) * dt;       % time array [s]

% MPC horizon
N_mpc      = 10;                     % prediction steps
dt_mpc     = dt;                     % MPC step = propagation step

% MPC cost weights
Q_diag = [10; 100; 50; 50; 50; 50];  % ROE tracking weights
% Higher weight on dlambda (along-track, unstable mode) and diy (science req)
Q_diag(2) = 200;   % dlambda: most critical — unstable drift
Q_diag(6) = 200;   % diy: determines observation geometry

Q  = diag(Q_diag);
R  = eye(3) * 1e6;    % control effort weight (high = fuel conscious)

% Initial conditions: science orbit with small perturbation
% (simulates acquisition error from Table 4: 5cm = 5e-2/ac)
rng(42);
sigma_ic  = 0.05 / ac;   % 5 cm acquisition error in ROE (dimensionless)
delta_alpha_0 = delta_alpha_star + sigma_ic * randn(6,1);

fprintf('Initial ROE error (vs target):\n');
fprintf('  [da, dlambda, dex, dey, dix, diy] * ac = ');
fprintf('%.3f  ', (delta_alpha_0 - delta_alpha_star)*ac);
fprintf('m\n\n');

% =========================================================================
% 5. CASADI SYMBOLIC DYNAMICS (discrete, RK4)
% =========================================================================
da_sym   = SX.sym('da',  6);    % ROE state
u_sym    = SX.sym('u',   3);    % EP control [m/s^2] in RTN
uc_sym   = SX.sym('uc',  1);    % mean argument of latitude

% Continuous ROE rate
xdot_expr = ROE_dynamics_J2_EP(da_sym, u_sym, uc_sym, auxdata);

f_roe = Function('f_roe', {da_sym, u_sym, uc_sym}, {xdot_expr});

% RK4 discrete step
k1 = f_roe(da_sym, u_sym, uc_sym);
k2 = f_roe(da_sym + dt_mpc/2*k1, u_sym, uc_sym + n*dt_mpc/2);
k3 = f_roe(da_sym + dt_mpc/2*k2, u_sym, uc_sym + n*dt_mpc/2);
k4 = f_roe(da_sym + dt_mpc*k3,   u_sym, uc_sym + n*dt_mpc);

da_next = da_sym + (dt_mpc/6)*(k1 + 2*k2 + 2*k3 + k4);

F_rk4 = Function('F_rk4', {da_sym, u_sym, uc_sym}, {da_next});

fprintf('CasADi RK4 ROE propagator ready (dt=%.0fs).\n\n', dt_mpc);

% =========================================================================
% 6. BUILD MPC OPTI PROBLEM (solved at each step)
% =========================================================================
% Pre-build the MPC Opti once — warm-start each solve
opti = casadi.Opti();

DA = opti.variable(6, N_mpc+1);   % ROE trajectory
U  = opti.variable(3, N_mpc);     % control sequence

p_x0  = opti.parameter(6, 1);     % current state
p_uc0 = opti.parameter(1, 1);     % current mean argument of latitude
p_ref = opti.parameter(6, 1);     % reference ROE

% Dynamics constraints
for k = 1:N_mpc
    uc_k   = p_uc0 + (k-1)*n*dt_mpc;
    da_pred = F_rk4(DA(:,k), U(:,k), uc_k);
    opti.subject_to( DA(:,k+1) == da_pred );
end

% Initial condition
opti.subject_to( DA(:,1) == p_x0 );

% Thrust magnitude bound (per-axis)
opti.subject_to( -a_max <= U <= a_max );

% Optional: total thrust bound ||u|| <= a_max
for k = 1:N_mpc
    opti.subject_to( U(1,k)^2 + U(2,k)^2 + U(3,k)^2 <= a_max^2 );
end

% MPC cost: tracking + fuel
cost = 0;
for k = 1:N_mpc
    err_k  = DA(:,k) - p_ref;
    cost   = cost + err_k' * Q * err_k + U(:,k)' * R * U(:,k);
end
err_N = DA(:,N_mpc+1) - p_ref;
cost  = cost + 10 * err_N' * Q * err_N;   % terminal cost

opti.minimize(cost);

% Quiet IPOPT
opts_mpc = struct();
opts_mpc.ipopt.print_level   = 0;
opts_mpc.ipopt.max_iter      = 200;
opts_mpc.ipopt.tol           = 1e-6;
opts_mpc.ipopt.warm_start_init_point = 'yes';
opts_mpc.print_time          = 0;
opti.solver('ipopt', opts_mpc);

% =========================================================================
% 7. CLOSED-LOOP MPC SIMULATION
% =========================================================================
fprintf('Running MPC simulation: %d orbits, %d steps...\n\n', N_orbits, N_total);

% Storage
X_hist  = zeros(6, N_total+1);
U_hist  = zeros(3, N_total);
dV_hist = zeros(1, N_total);
uc_hist = zeros(1, N_total+1);

X_hist(:,1)  = delta_alpha_0;
uc_hist(1)   = 0;

% Warm-start values
U_prev  = zeros(3, N_mpc);
X_prev  = repmat(delta_alpha_0, 1, N_mpc+1);

total_dV     = 0;
mpc_failures = 0;

t_start = tic;

for k = 1:N_total
    x_k  = X_hist(:,k);
    uc_k = uc_hist(k);

    % --- Set MPC parameters ---
    opti.set_value(p_x0,  x_k);
    opti.set_value(p_uc0, uc_k);
    opti.set_value(p_ref, delta_alpha_star);

    % --- Warm start ---
    opti.set_initial(DA, X_prev);
    opti.set_initial(U,  [U_prev(:,2:end), zeros(3,1)]);

    % --- Solve ---
    try
        sol = opti.solve();
        U_opt   = sol.value(U);
        X_opt   = sol.value(DA);
        U_prev  = U_opt;
        X_prev  = X_opt;
    catch
        % Fallback: zero control if MPC fails
        U_opt   = zeros(3, N_mpc);
        mpc_failures = mpc_failures + 1;
    end

    % Apply first control
    u_apply = U_opt(:,1);
    U_hist(:,k) = u_apply;

    % Propagate true state one step
    x_next    = full(F_rk4(x_k, u_apply, uc_k));
    X_hist(:,k+1) = x_next;
    uc_hist(k+1)  = uc_k + n * dt;

    % Accumulate delta-V
    dv_k        = norm(u_apply) * dt;   % [m/s]
    dV_hist(k)  = dv_k;
    total_dV    = total_dV + dv_k;

    % Progress
    if mod(k, round(N_total/10)) == 0
        err_norm = norm((x_k - delta_alpha_star) * ac);
        fprintf('  t = %.1f hr | ||ROE_err|| = %.3f m | cumΔV = %.4f mm/s\n', ...
                t_vec(k)/3600, err_norm, total_dV*1e3);
    end
end

sim_time = toc(t_start);
fprintf('\nSimulation complete: %.1f s (MPC failures: %d)\n\n', sim_time, mpc_failures);

% =========================================================================
% 8. COMPUTE EQUIVALENT DELAY LINE (Eq. 7 in STARI paper)
%    D = delta_r . s_hat (dot product of relative position with star dir)
%    For science orbit condition, this should stay near zero.
% =========================================================================
phi0 = phi0_deg * pi/180;
beta = beta_deg * pi/180;

delay_line = zeros(1, N_total+1);
baseline   = zeros(1, N_total+1);

for k = 1:N_total+1
    uc_k   = uc_hist(k);
    da_k   = X_hist(:,k);

    % RTN relative position from ROE (Eq. 3)
    dr_R = da_k(1) - cos(uc_k)*da_k(3) - sin(uc_k)*da_k(4);
    dr_T = da_k(2) + 2*sin(uc_k)*da_k(3) - 2*cos(uc_k)*da_k(4);
    dr_N = sin(uc_k)*da_k(5) - cos(uc_k)*da_k(6);

    dr_R = dr_R * ac;   % [m]
    dr_T = dr_T * ac;
    dr_N = dr_N * ac;

    % Star direction in RTN (Eq. 4)
    theta = phi0 - uc_k;
    s_R   = cos(beta)*cos(theta);
    s_T   = cos(beta)*sin(theta);
    s_N   = sin(beta);

    % Delay line = dot product
    delay_line(k) = dr_R*s_R + dr_T*s_T + dr_N*s_N;

    % Baseline = cross product magnitude (perpendicular separation)
    dr_cross_s = [dr_T*s_N - dr_N*s_T;
                  dr_N*s_R - dr_R*s_N;
                  dr_R*s_T - dr_T*s_R];
    baseline(k) = norm(dr_cross_s);
end

% =========================================================================
% 9. RESULTS SUMMARY
% =========================================================================
t_orbits = t_vec / T_orb;   % time in orbital periods

ROE_err_m = (X_hist - delta_alpha_star) * ac;  % ROE error in meters

fprintf('==================== RESULTS ====================\n');
fprintf('Total simulation: %.1f orbits  (%.1f hr)\n', N_orbits, t_vec(end)/3600);
fprintf('\nROE tracking error (RMS, meters):\n');
labels = {'  da    ','  dlambda','  dex   ','  dey   ','  dix   ','  diy   '};
for i = 1:6
    fprintf('%s = %.4f m\n', labels{i}, rms(ROE_err_m(i,:)));
end

fprintf('\nDelay line (REQ.3: D < 0.5 m):\n');
fprintf('  Max |D|      = %.4f m\n', max(abs(delay_line)));
fprintf('  RMS |D|      = %.4f m\n', rms(delay_line));
frac_ok = mean(abs(delay_line) < 0.5);
fprintf('  %% within 0.5m = %.1f%% (REQ.3 %s)\n', frac_ok*100, ...
        ifelse(frac_ok>0.8,'PASS','marginal'));

fprintf('\nBaseline (REQ.1: 10-150 m):\n');
fprintf('  Mean B = %.2f m\n', mean(baseline));
fprintf('  Min B  = %.2f m\n', min(baseline));
fprintf('  Max B  = %.2f m\n', max(baseline));

fprintf('\nFuel budget:\n');
fprintf('  Total ΔV = %.4f mm/s\n', total_dV*1e3);
fprintf('  Per orbit = %.4f mm/s\n', total_dV*1e3/N_orbits);
fprintf('  Paper (S1 SK mean): 0.076 mm/s per orbit\n');

% =========================================================================
% 10. PLOTS
% =========================================================================

% --- Figure 1: ROE history ---
figure('Name','ROE History','Position',[50 50 1200 800]);
roe_labels = {'\delta a [m]','\delta\lambda [m]','\delta e_x [m]',...
              '\delta e_y [m]','\delta i_x [m]','\delta i_y [m]'};
roe_targets = delta_alpha_star * ac;
for i = 1:6
    subplot(3,2,i)
    plot(t_orbits, X_hist(i,:)*ac, 'b-', 'LineWidth',1.2); hold on
    yline(roe_targets(i), 'r--', 'LineWidth',1.5)
    xlabel('Time [orbits]'); ylabel(roe_labels{i})
    title(roe_labels{i}); grid on
    legend('Actual','Target','Location','best')
end
sgtitle('ROE State History with EP Station Keeping', 'FontSize',13)

% --- Figure 2: Delay line ---
figure('Name','Delay Line','Position',[50 50 900 400]);
plot(t_orbits, delay_line, 'k-', 'LineWidth',1.2); hold on
yline(0.5,  'r--', 'LineWidth',1.5)
yline(-0.5, 'r--', 'LineWidth',1.5)
xlabel('Time [orbits]'); ylabel('Delay Line D [m]')
title('Equivalent Delay Line (REQ.3: |D| < 0.5 m)')
legend('D(t)', 'REQ.3 limit', 'Location','best')
grid on
ylim([-1 1])

% --- Figure 3: Control history ---
figure('Name','EP Control','Position',[50 50 900 400]);
u_labels = {'u_R [m/s^2]','u_T [m/s^2]','u_N [m/s^2]'};
for i = 1:3
    subplot(1,3,i)
    stairs(t_orbits(1:N_total), U_hist(i,:), 'r-', 'LineWidth',1.0)
    yline(a_max, 'b--'); yline(-a_max, 'b--')
    xlabel('Time [orbits]'); ylabel(u_labels{i})
    title(u_labels{i}); grid on
end
sgtitle('EP Control History (RTN Components)', 'FontSize',13)

% --- Figure 4: Relative motion in RTN ---
figure('Name','Relative Motion RTN','Position',[100 100 1000 400]);
dr_R_hist = (X_hist(1,:) - cos(uc_hist).*X_hist(3,:) - sin(uc_hist).*X_hist(4,:)) * ac;
dr_T_hist = (X_hist(2,:) + 2*sin(uc_hist).*X_hist(3,:) - 2*cos(uc_hist).*X_hist(4,:)) * ac;
dr_N_hist = (sin(uc_hist).*X_hist(5,:) - cos(uc_hist).*X_hist(6,:)) * ac;

subplot(1,3,1)
plot(dr_T_hist, dr_R_hist, 'b-', 'LineWidth',0.8)
xlabel('T [m]'); ylabel('R [m]'); title('R-T Plane'); grid on; axis equal

subplot(1,3,2)
plot(dr_T_hist, dr_N_hist, 'b-', 'LineWidth',0.8)
xlabel('T [m]'); ylabel('N [m]'); title('T-N Plane'); grid on; axis equal

subplot(1,3,3)
plot(dr_R_hist, dr_N_hist, 'b-', 'LineWidth',0.8)
xlabel('R [m]'); ylabel('N [m]'); title('R-N Plane'); grid on; axis equal

sgtitle('Relative Motion in RTN Frame', 'FontSize',13)

% --- Figure 5: Relative inclination plane (Fig. 6 in STARI paper) ---
figure('Name','Inclination Plane','Position',[150 150 600 550]);
plot(X_hist(5,:)*ac, X_hist(6,:)*ac, 'b-', 'LineWidth',1.0); hold on
plot(delta_alpha_star(5)*ac, delta_alpha_star(6)*ac, 'r*', 'MarkerSize',15, 'LineWidth',2)
plot(X_hist(5,1)*ac, X_hist(6,1)*ac, 'go', 'MarkerSize',10, 'MarkerFaceColor','g')

% Plot the circle of valid science orbits
theta_circle = linspace(0,2*pi,200);
r_circle     = delta_lambda_target * ac / tan(beta);
plot(r_circle*cos(theta_circle), r_circle*sin(theta_circle), 'k--', 'LineWidth',1)

xlabel('a\deltai_x [m]'); ylabel('a\deltai_y [m]')
title('Relative Inclination Plane')
legend('Trajectory','Target (phi_0=90)','Initial','Valid science circle','Location','best')
grid on; axis equal

% --- Figure 6: Baseline history ---
figure('Name','Baseline','Position',[200 200 800 350]);
subplot(1,2,1)
plot(t_orbits, baseline, 'b-', 'LineWidth',1.2); hold on
yline(orbit_info.B_min, 'r--'); yline(orbit_info.B_max, 'r--')
yline(10, 'k:'); yline(150, 'k:')
xlabel('Time [orbits]'); ylabel('Baseline B [m]')
title('Interferometric Baseline'); grid on
legend('B(t)','B_{min/max} analytic','REQ.1 limits','Location','best')

subplot(1,2,2)
cumDV_mmps = cumsum(dV_hist)*1e3;
plot(t_orbits(1:N_total), cumDV_mmps, 'r-', 'LineWidth',1.5)
xlabel('Time [orbits]'); ylabel('\DeltaV [mm/s]')
title('Cumulative \DeltaV'); grid on

sgtitle('Baseline and Fuel Budget', 'FontSize',13)

% =========================================================================
function out = ifelse(c,a,b)
if c; out=a; else; out=b; end
end
