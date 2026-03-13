% =========================================================================
% lowThrust_OCP_CasADi.m  (v3)
%
% Low-Thrust Orbit Transfer OCP — CasADi / IPOPT
% LEO (i=28.5 deg)  ->  HEO (i=63.4 deg),  maximize final mass
%
%
% Dependencies: lowThrust_MEE_dynamics.m, lowThrust_initialGuess.m
% =========================================================================

clc; clear; close all;
% Get the full path of the current file
currentFilePath = mfilename('fullpath');

% Get the folder containing this file
[currentFolder, ~, ~] = fileparts(currentFilePath);

% Go up one directory
parentFolder = fileparts(currentFolder);

% Example: build a path to a file in the parent directory
addpath(genpath(fullfile(parentFolder,'casadi-3.7.2-windows64-matlab2018b/')))

import casadi.*

fprintf('===========================================\n');
fprintf(' Low-Thrust OCP  (CasADi v3, scaled NLP)\n');
fprintf('===========================================\n\n');

% =========================================================================
% 1. PHYSICAL CONSTANTS (SI)
% =========================================================================
ft2m   = 0.3048;
lbf2N  = 4.44822;
lbm2kg = 0.453592;

auxdata.mu  = 3.9860e14;   %   m^3/s^2
auxdata.T   = 0.01978;       %     N
auxdata.Isp = 450;                         % s
auxdata.Re  = 6.3781e6;         %    m
auxdata.J2  = 1082.639e-6;
auxdata.J3  = -2.565e-6;
auxdata.J4  = -1.608e-6;
auxdata.p0       =  6.6559e6;  %  m
auxdata.h0       = -0.25396764647494;
auxdata.w0       =  0.4536;            %    kg
auxdata.tf_guess = 90000;                    % s

pf_tgt  = 12194239.065442712977;
ecc_tgt = 0.73550320568829;
inc_tgt = 0.61761258786099;

% =========================================================================
% 2. SCALING  (all scaled variables are ~O(1))
% =========================================================================
% Raw state:   x = [p[m], f, g, h, k, L[rad], w[kg]]
% Scaled:      xs = [p/p_s, f, g, h, k, L/L_s, w/w_s]
%
% Raw time:    t [s]
% Scaled time: tau_t in [0,1],  t = tau_t * tf
%
% Control u=[ur,ut,uh] is already O(1) — no scaling needed.

p_s = auxdata.p0;            % ~6.66e6 m
L_s = 9*2*pi;                % max expected true longitude ~9 revs
w_s = auxdata.w0;            % 0.4536 kg
tf_s = auxdata.tf_guess;     % 90000 s

% Build scaling vector for 7 states
S = [p_s; 1; 1; 1; 1; L_s; w_s];   % element-wise state scale

fprintf('State scales: p=%.2e  L=%.2f  w=%.4f\n\n', p_s, L_s, w_s);

% =========================================================================
% 3. GRID
% =========================================================================
N = 150;
fprintf('N = %d shooting intervals\n\n', N);

% =========================================================================
% 4. INITIAL GUESS
% =========================================================================
guess = lowThrust_initialGuess(auxdata, N);

% Scale the guessed states
Xg_scaled = guess.X ./ S';    % [N+1 x 7], each row divided by S'

% =========================================================================
% 5. CASADI SCALED DYNAMICS
%    Integrate xs_dot = (tf/S) .* f_raw(S.*xs, u, tau)
% =========================================================================
xs_sym  = SX.sym('xs',  7);   % scaled state
u_sym   = SX.sym('u',   3);   % control (unscaled, O(1))
tau_sym = SX.sym('tau', 1);   % throttle
tf_sym  = SX.sym('tf',  1);   % final time [s] (unscaled)

% Recover physical state from scaled
x_phys = xs_sym .* S;

% Raw dynamics in physical units
xdot_raw = lowThrust_MEE_dynamics(x_phys, u_sym, tau_sym, auxdata);

% Scale derivatives + normalize time to [0,1]
xsdot = tf_sym * (xdot_raw ./ S);

f_scaled = Function('f_scaled', {xs_sym, u_sym, tau_sym, tf_sym}, {xsdot});

% RK4 on [0,1]
h = 1/N;
k1 = f_scaled(xs_sym, u_sym, tau_sym, tf_sym);
k2 = f_scaled(xs_sym + (h/2)*k1, u_sym, tau_sym, tf_sym);
k3 = f_scaled(xs_sym + (h/2)*k2, u_sym, tau_sym, tf_sym);
k4 = f_scaled(xs_sym + h*k3,     u_sym, tau_sym, tf_sym);
xs_next = xs_sym + (h/6)*(k1 + 2*k2 + 2*k3 + k4);

F_rk4 = Function('F_rk4', {xs_sym, u_sym, tau_sym, tf_sym}, {xs_next});

fprintf('Scaled CasADi RK4 ready.\n\n');

% =========================================================================
% 6. BUILD OPTI
% =========================================================================
opti = casadi.Opti();

XS  = opti.variable(7, N+1);   % SCALED states
U   = opti.variable(3, N);     % controls
tau = opti.variable(1, 1);     % throttle
tf  = opti.variable(1, 1);     % final time [s]

% =========================================================================
% 7. DYNAMICS CONSTRAINTS (in scaled coordinates)
% =========================================================================
for k = 1:N
    xs_pred = F_rk4(XS(:,k), U(:,k), tau, tf);
    opti.subject_to( XS(:,k+1) == xs_pred );
end

% =========================================================================
% 8. PATH CONSTRAINT: unit thrust vector (relaxed bracket)
% =========================================================================
for k = 1:N
    nrm2 = U(1,k)^2 + U(2,k)^2 + U(3,k)^2;
    opti.subject_to( 0.98 <= nrm2 <= 1.02 );
end

% =========================================================================
% 9. BOUNDS (scaled where applicable)
% =========================================================================
opti.subject_to( -1   <= U   <= 1   );
opti.subject_to( -50  <= tau <= 0   );
opti.subject_to( 50000 <= tf <= 120000 );

% Scaled state bounds
p_min_s = (20e6*ft2m) / p_s;
p_max_s = (60e6*ft2m) / p_s;
L_min_s = pi / L_s;

opti.subject_to( p_min_s <= XS(1,:) <= p_max_s );
opti.subject_to( -1      <= XS(2,:) <= 1       );
opti.subject_to( -1      <= XS(3,:) <= 1       );
opti.subject_to( -1      <= XS(4,:) <= 1       );
opti.subject_to( -1      <= XS(5,:) <= 1       );
opti.subject_to( L_min_s <= XS(6,:)            );
opti.subject_to( 0.005/w_s <= XS(7,:) <= 1.0  );  % mass in (0, w0]

% =========================================================================
% 10. BOUNDARY CONDITIONS (in scaled coordinates)
% =========================================================================
opti.subject_to( XS(1,1) == auxdata.p0 / p_s );
opti.subject_to( XS(2,1) == 0                );
opti.subject_to( XS(3,1) == 0                );
opti.subject_to( XS(4,1) == auxdata.h0       );
opti.subject_to( XS(5,1) == 0                );
opti.subject_to( XS(6,1) == pi / L_s         );
opti.subject_to( XS(7,1) == auxdata.w0 / w_s );  % = 1.0 by construction

% Terminal — unscale p before comparing
pf_s = pf_tgt / p_s;
opti.subject_to( XS(1,N+1) == pf_s );

% f,g,h,k are unscaled, use directly
ff = XS(2,N+1);  gf = XS(3,N+1);
hf = XS(4,N+1);  kf = XS(5,N+1);

opti.subject_to( ff^2 + gf^2 == ecc_tgt^2 );
opti.subject_to( hf^2 + kf^2 == inc_tgt^2 );
opti.subject_to( ff*hf + gf*kf == 0        );
opti.subject_to( gf*hf - kf*ff <= 0        );

% =========================================================================
% 11. OBJECTIVE: maximize final (scaled) mass
% =========================================================================
opti.minimize( -XS(7, N+1) );

% =========================================================================
% 12. INITIAL GUESS
% =========================================================================
opti.set_initial(XS,  Xg_scaled');
opti.set_initial(U,   guess.U(1:N,:)');
opti.set_initial(tau, -25);
opti.set_initial(tf,  auxdata.tf_guess);

% =========================================================================
% 13. IPOPT OPTIONS
%     Use L-BFGS (limited-memory) Hessian approximation — much faster
%     per iteration for large problems. Switch to 'exact' for final solve.
% =========================================================================
opts_fast = struct();
opts_fast.ipopt.max_iter              = 2000;
opts_fast.ipopt.tol                   = 1e-4;
opts_fast.ipopt.acceptable_tol        = 1e-3;
opts_fast.ipopt.acceptable_iter       = 5;
opts_fast.ipopt.print_level           = 5;
opts_fast.ipopt.mu_strategy           = 'adaptive';
opts_fast.ipopt.nlp_scaling_method    = 'none';  % we scale manually
opts_fast.ipopt.linear_solver         = 'mumps';
opts_fast.ipopt.hessian_approximation = 'limited-memory';  % L-BFGS
opts_fast.ipopt.limited_memory_max_history = 10;

opti.solver('ipopt', opts_fast);

% =========================================================================
% 14. SOLVE (Stage 1: L-BFGS, relaxed tolerance)
% =========================================================================
fprintf('--- Stage 1: L-BFGS solve (fast, relaxed tol) ---\n\n');
try
    tic;
    sol1 = opti.solve();
    t1 = toc;
    fprintf('\nStage 1 converged in %.1f s.\n\n', t1);
    x1 = sol1.value(XS);
    u1 = sol1.value(U);
    tau1 = sol1.value(tau);
    tf1  = sol1.value(tf);
    stage1_ok = true;
catch ME
    fprintf('\nStage 1 message: %s\n', ME.message);
    sol1 = opti.debug;
    x1   = sol1.value(XS);
    u1   = sol1.value(U);
    tau1 = sol1.value(tau);
    tf1  = sol1.value(tf);
    stage1_ok = false;
end

report_solution(x1, u1, tau1, tf1, N, S, pf_tgt, ecc_tgt, inc_tgt, stage1_ok, 1);

% =========================================================================
% 15. SOLVE (Stage 2: exact Hessian, tight tolerance, warm start)
%     Only run if Stage 1 produced a reasonable iterate
% =========================================================================
if true   % set to false to skip if Stage 1 didn't converge
    fprintf('\n--- Stage 2: Exact Hessian, tight tolerance ---\n\n');

    opti2 = casadi.Opti();

    XS2  = opti2.variable(7, N+1);
    U2   = opti2.variable(3, N);
    tau2 = opti2.variable(1,1);
    tf2  = opti2.variable(1,1);

    % Same constraints — rebuild for clean warm start
    for k = 1:N
        xs_pred = F_rk4(XS2(:,k), U2(:,k), tau2, tf2);
        opti2.subject_to( XS2(:,k+1) == xs_pred );
    end
    for k = 1:N
        nrm2 = U2(1,k)^2 + U2(2,k)^2 + U2(3,k)^2;
        opti2.subject_to( 0.999 <= nrm2 <= 1.001 );  % tighter in Stage 2
    end
    opti2.subject_to( -1  <= U2  <= 1  );
    opti2.subject_to( -50 <= tau2 <= 0 );
    opti2.subject_to( 50000 <= tf2 <= 120000 );
    opti2.subject_to( p_min_s <= XS2(1,:) <= p_max_s );
    opti2.subject_to( -1 <= XS2(2,:) <= 1 );
    opti2.subject_to( -1 <= XS2(3,:) <= 1 );
    opti2.subject_to( -1 <= XS2(4,:) <= 1 );
    opti2.subject_to( -1 <= XS2(5,:) <= 1 );
    opti2.subject_to( L_min_s <= XS2(6,:) );
    opti2.subject_to( 0.005/w_s <= XS2(7,:) <= 1.0 );
    opti2.subject_to( XS2(1,1) == auxdata.p0/p_s );
    opti2.subject_to( XS2(2,1) == 0             );
    opti2.subject_to( XS2(3,1) == 0             );
    opti2.subject_to( XS2(4,1) == auxdata.h0    );
    opti2.subject_to( XS2(5,1) == 0             );
    opti2.subject_to( XS2(6,1) == pi/L_s        );
    opti2.subject_to( XS2(7,1) == 1.0           );
    opti2.subject_to( XS2(1,N+1) == pf_s        );
    ff2=XS2(2,N+1); gf2=XS2(3,N+1); hf2=XS2(4,N+1); kf2=XS2(5,N+1);
    opti2.subject_to( ff2^2+gf2^2 == ecc_tgt^2  );
    opti2.subject_to( hf2^2+kf2^2 == inc_tgt^2  );
    opti2.subject_to( ff2*hf2+gf2*kf2 == 0      );
    opti2.subject_to( gf2*hf2-kf2*ff2 <= 0      );
    opti2.minimize( -XS2(7,N+1) );

    % Warm start from Stage 1
    opti2.set_initial(XS2,  x1);
    opti2.set_initial(U2,   u1);
    opti2.set_initial(tau2, tau1);
    opti2.set_initial(tf2,  tf1);

    opts_tight = struct();
    opts_tight.ipopt.max_iter           = 1000;
    opts_tight.ipopt.tol                = 1e-6;
    opts_tight.ipopt.print_level        = 5;
    opts_tight.ipopt.mu_strategy        = 'adaptive';
    opts_tight.ipopt.nlp_scaling_method = 'none';
    opts_tight.ipopt.linear_solver      = 'mumps';
    opts_tight.ipopt.warm_start_init_point = 'yes';

    opti2.solver('ipopt', opts_tight);

    try
        tic;
        sol2 = opti2.solve();
        t2 = toc;
        fprintf('\nStage 2 converged in %.1f s.\n\n', t2);
        x2 = sol2.value(XS2); u2 = sol2.value(U2);
        tau2v = sol2.value(tau2); tf2v = sol2.value(tf2);
        stage2_ok = true;
    catch ME2
        fprintf('\nStage 2 message: %s\n', ME2.message);
        sol2 = opti2.debug;
        x2 = sol2.value(XS2); u2 = sol2.value(U2);
        tau2v = sol2.value(tau2); tf2v = sol2.value(tf2);
        stage2_ok = false;
    end

    report_solution(x2, u2, tau2v, tf2v, N, S, pf_tgt, ecc_tgt, inc_tgt, stage2_ok, 2);
    plot_solution(x2, u2, tf2v, N, S, auxdata);
else
    plot_solution(x1, u1, tf1, N, S, auxdata);
end

% =========================================================================
% HELPER: report
% =========================================================================
function report_solution(XS, U, tau, tf, N, S, pf_tgt, ecc_tgt, inc_tgt, ok, stage)
w_s = S(7); p_s = S(1); L_s = S(6);
fprintf('--- Stage %d Results (%s) ---\n', stage, ifelse(ok,'OPTIMAL','best iterate'));
fprintf('  tf           = %.1f s (%.3f hr)\n', tf, tf/3600);
fprintf('  tau          = %.4f\n', tau);
fprintf('  m_initial    = %.6f kg\n', XS(7,1)  * w_s);
fprintf('  m_final      = %.6f kg\n', XS(7,end)* w_s);
fprintf('  fuel used    = %.6f kg\n', (XS(7,1)-XS(7,end))*w_s);

ff=XS(2,end); gf=XS(3,end); hf=XS(4,end); kf=XS(5,end);
pf_act = XS(1,end)*p_s;
fprintf('  p error      = %+.4e m\n',  pf_act - pf_tgt);
fprintf('  ecc residual = %+.4e\n',    sqrt(ff^2+gf^2)-ecc_tgt);
fprintf('  inc residual = %+.4e\n',    sqrt(hf^2+kf^2)-inc_tgt);
fprintf('  cross term   = %+.4e\n',    ff*hf+gf*kf);
u_nrm = sqrt(sum(U.^2,1));
fprintf('  ||u|| range  = [%.6f, %.6f]\n\n', min(u_nrm), max(u_nrm));
end

% =========================================================================
% HELPER: plots
% =========================================================================
function plot_solution(XS, U, tf, N, S, auxdata)
p_s=S(1); L_s=S(6); w_s=S(7);
t_hr = linspace(0, tf/3600, N+1);

% Unscale
X_phys = XS .* S;

figure('Name','State History','Position',[50 50 1200 750]);
labs = {'p [m]','f','g','h','k','L [rad]','mass [kg]'};
for i=1:7
    subplot(3,3,i)
    plot(t_hr, X_phys(i,:), 'b-', 'LineWidth',1.5)
    xlabel('Time [hr]'); ylabel(labs{i}); title(labs{i}); grid on
end
sgtitle('MEE State History', 'FontSize',13)

figure('Name','Control History','Position',[50 50 900 350]);
clabs = {'u_r','u_\theta','u_h'};
for i=1:3
    subplot(1,3,i)
    plot(linspace(0,tf/3600,N), U(i,:), 'r-', 'LineWidth',1.5)
    xlabel('Time [hr]'); ylabel(clabs{i}); ylim([-1.1 1.1]); grid on
end
sgtitle('Control History', 'FontSize',13)

% 3D trajectory
figure('Name','3D Trajectory','Position',[100 100 750 650]);
[rX,rY,rZ] = mee2eci(X_phys);
plot3(rX/1e6, rY/1e6, rZ/1e6, 'r-', 'LineWidth',1.5); hold on
plot3(rX(1)/1e6,   rY(1)/1e6,   rZ(1)/1e6,   'go','MarkerSize',10,'MarkerFaceColor','g')
plot3(rX(end)/1e6, rY(end)/1e6, rZ(end)/1e6, 'rs','MarkerSize',10,'MarkerFaceColor','r')
xlabel('x [Mm]'); ylabel('y [Mm]'); zlabel('z [Mm]')
title('Optimal Low-Thrust Transfer'); grid on; axis equal
legend('Trajectory','Initial','Final')
end

function [rX,rY,rZ] = mee2eci(X)
n=size(X,2); rX=zeros(1,n); rY=rX; rZ=rX;
for i=1:n
    p=X(1,i);f=X(2,i);g=X(3,i);h=X(4,i);k=X(5,i);L=X(6,i);
    q=1+f*cos(L)+g*sin(L); r=p/q;
    s2=1+h^2+k^2; a2=h^2-k^2;
    rX(i)=(r/s2)*(cos(L)+a2*cos(L)+2*h*k*sin(L));
    rY(i)=(r/s2)*(sin(L)-a2*sin(L)+2*h*k*cos(L));
    rZ(i)=(2*r/s2)*(h*sin(L)-k*cos(L));
end
end

function out = ifelse(c,a,b)
if c; out=a; else; out=b; end
end
