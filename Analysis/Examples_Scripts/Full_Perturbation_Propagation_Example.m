clear; clc;

mu = 398600;          % km^3/s^2
RE = 6378;            % km
J2 = 1082.63e-6;

params.mu      = mu;
params.RE      = RE;
params.J2      = J2;
params.muMoon  = 4903;          % km^3/s^2
params.muSun   = 132.712e9;     % km^3/s^2
params.CR      = 2;
params.As      = 4.0;           % m^2
params.m       = 100;           % kg
params.S       = 1367;          % W/m^2
params.c       = 2.998e8;       % m/s
params.useShadow = true;
params.ephemModel = '421';


deg = pi/180;
params.jd0     = juliandate(datetime(2024,1,1,0,0,0));


% Example near-circular initial state
a0    = RE + 35786;      % km
e0    = 1e-4;
inc0  = 7*deg;
RAAN0 = 0;
w0    = 0;
M0    = 0;
u0    = w0 + M0;

ex0 = e0*cos(w0);
ey0 = e0*sin(w0);

x0 = [a0; ex0; ey0; inc0; RAAN0; u0];

T0 = 2*pi*sqrt(a0^3/mu);

t0 = 0;
tf = 24*3600;

tspan = t0:30*60:tf;
params.ephem = precompute_ephemeris(tspan, params);

opts = odeset('RelTol',1e-6,'AbsTol',1e-6);

% Propagate total state
[t,x] = ode45(@(t,x) rates_qns_total(t,x,params), tspan, x0, opts);

% Allocate breakdown arrays
N = length(t);

rates_total = zeros(N,6);
rates_J2    = zeros(N,6);
rates_SRP   = zeros(N,6);
rates_Moon  = zeros(N,6);
rates_Sun   = zeros(N,6);

acc_J2   = zeros(N,3);
acc_SRP  = zeros(N,3);
acc_Moon = zeros(N,3);
acc_Sun  = zeros(N,3);

for k = 1:N
    tmp = qns_perturbation_breakdown(t(k), x(k,:).', params);

    rates_total(k,:) = tmp.rates.total.';
    rates_J2(k,:)    = tmp.rates.J2.';
    rates_SRP(k,:)   = tmp.rates.SRP.';
    rates_Moon(k,:)  = tmp.rates.Moon.';
    rates_Sun(k,:)   = tmp.rates.Sun.';

    acc_J2(k,:)   = tmp.accel.J2.';
    acc_SRP(k,:)  = tmp.accel.SRP.';
    acc_Moon(k,:) = tmp.accel.Moon.';
    acc_Sun(k,:)  = tmp.accel.Sun.';
end

% State history
a    = x(:,1);
ex   = x(:,2);
ey   = x(:,3);
inc  = x(:,4);
RAAN = x(:,5);
u    = x(:,6);
e    = sqrt(ex.^2 + ey.^2);

%% Plot total state history
figure;
subplot(3,2,1); plot(t/86400,a-a0); title('a-a_0 (km)'); grid on
subplot(3,2,2); plot(t/86400,e-e0); title('e-e_0'); grid on
subplot(3,2,3); plot(t/86400,(inc-inc0)/deg); title('i-i_0 (deg)'); grid on
subplot(3,2,4); plot(t/86400,unwrap(RAAN-RAAN0)/deg); title('\Omega-\Omega_0 (deg)'); grid on
subplot(3,2,5); plot(t/86400,unwrap(u-u0)/deg); title('u-u_0 (deg)'); grid on

%% Element-rate breakdown plots
labels = {'a dot','e_x dot','e_y dot','i dot','\Omega dot','u dot'};

for j = 1:6
    figure;
    plot(t/86400, rates_J2(:,j),  'LineWidth',1.2); hold on
    plot(t/86400, rates_SRP(:,j), 'LineWidth',1.2);
    plot(t/86400, rates_Moon(:,j),'LineWidth',1.2);
    plot(t/86400, rates_Sun(:,j), 'LineWidth',1.2);
    plot(t/86400, rates_total(:,j),'k--','LineWidth',1.4);
    xlabel('Time (days)');
    ylabel(labels{j});
    title(['Perturbation Breakdown: ', labels{j}]);
    legend('J2','SRP','Moon','Sun grav','Total');
    grid on
end

%% RTN acceleration magnitude breakdown
accmag_J2   = vecnorm(acc_J2,2,2);
accmag_SRP  = vecnorm(acc_SRP,2,2);
accmag_Moon = vecnorm(acc_Moon,2,2);
accmag_Sun  = vecnorm(acc_Sun,2,2);

figure;
plot(t/86400, accmag_J2,   'LineWidth',1.2); hold on
plot(t/86400, accmag_SRP,  'LineWidth',1.2);
plot(t/86400, accmag_Moon, 'LineWidth',1.2);
plot(t/86400, accmag_Sun,  'LineWidth',1.2);
xlabel('Time (days)');
ylabel('|a_p| (km/s^2)');
title('Perturbing Acceleration Magnitude Breakdown');
legend('J2','SRP','Moon','Sun grav');
grid on