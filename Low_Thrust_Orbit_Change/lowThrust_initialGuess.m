function guess = lowThrust_initialGuess(auxdata, N)
% =========================================================================
% lowThrust_initialGuess.m  (v3)
%
%
% The OCP solver will then find the correct throttle/fuel tradeoff.
% =========================================================================

fprintf('Generating initial guess (v3 — low-tau propagation)...\n');

x0 = [auxdata.p0; 0; 0; auxdata.h0; 0; pi; auxdata.w0];

% tau = -25 gives T_eff = 0.75*T
% With correct mdot = T/(Isp*g0), burn time ~28 hr > 25 hr transfer — mass survives.
tau_guess = -25;
tf_prop   = auxdata.tf_guess;   % full 90,000 s

ode_fun = @(t,x) prop_ode(t, x, tau_guess, auxdata);

opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-11, 'MaxStep', 300);
fprintf('  Propagating %.0f s with tau=%.1f (%.1f%% thrust)...\n', ...
        tf_prop, tau_guess, (1+0.01*tau_guess)*100);

[t_p, X_p] = ode15s(ode_fun, [0, tf_prop], x0, opts);

fprintf('  Done: %d steps, tf=%.0f s (%.2f hr)\n', ...
        length(t_p), t_p(end), t_p(end)/3600);
fprintf('  Final mass: %.6f kg (initial: %.6f kg)\n', X_p(end,7), x0(7));
fprintf('  Final p:    %.4e m\n', X_p(end,1));
fprintf('  Final L:    %.2f rad (%.1f revs)\n', X_p(end,6), X_p(end,6)/(2*pi));

% --- Interpolate onto N+1 uniform grid ---
t_grid = linspace(0, tf_prop, N+1)';
X_grid = zeros(N+1, 7);
for i = 1:7
    X_grid(:,i) = interp1(t_p, X_p(:,i), t_grid, 'pchip');
end

% Hard-clamp mass to stay positive
X_grid(:,7) = max(X_grid(:,7), 0.01*auxdata.w0);

% --- Velocity-aligned control ---
U_grid = zeros(N+1, 3);
for i = 1:N+1
    U_grid(i,:) = vel_aligned_ctrl(X_grid(i,:)', auxdata)';
end

guess.t  = t_grid;
guess.X  = X_grid;
guess.U  = U_grid;
guess.tf = tf_prop;

fprintf('  Interpolated to %d grid points.\n\n', N+1);
end

function xdot = prop_ode(~, x, tau, auxdata)
u = vel_aligned_ctrl(x, auxdata);
xdot = lowThrust_MEE_dynamics(x, u, tau, auxdata);
end

function u_rtn = vel_aligned_ctrl(x, auxdata)
mu = auxdata.mu;
p=x(1); f=x(2); g=x(3); h=x(4); k=x(5); L=x(6);
s2=1+h^2+k^2; alpha2=h^2-k^2;
q=1+f*cos(L)+g*sin(L); r=p/q;

rX=(r/s2)*(cos(L)+alpha2*cos(L)+2*h*k*sin(L));
rY=(r/s2)*(sin(L)-alpha2*sin(L)+2*h*k*cos(L));
rZ=(2*r/s2)*(h*sin(L)-k*cos(L));
r_ECI=[rX;rY;rZ]; rMag=norm(r_ECI);

vX=-(1/s2)*sqrt(mu/p)*(sin(L)+alpha2*sin(L)-2*h*k*cos(L)+g-2*f*h*k+alpha2*g);
vY=-(1/s2)*sqrt(mu/p)*(-cos(L)+alpha2*cos(L)+2*h*k*sin(L)-f+2*g*h*k+alpha2*f);
vZ= (2/s2)*sqrt(mu/p)*(h*cos(L)+k*sin(L)+f*h+g*k);
v_ECI=[vX;vY;vZ];

ir=r_ECI/rMag;
hv=cross(r_ECI,v_ECI); hv=hv/norm(hv);
it=cross(hv,ir);

vhat=v_ECI/norm(v_ECI);
u_rtn=[dot(vhat,ir); dot(vhat,it); dot(vhat,hv)];
u_rtn=u_rtn/norm(u_rtn);
end
