
mu = 398600;          % km^3/s^2
RE = 6378;            % km
J2 = 1082.63e-6;
deg = pi/180;

%% ------------------------------------------------------------------------
% Chief orbit (GEO, frozen)
%% ------------------------------------------------------------------------
a0    = 42164;               % km
e0    = 1e-4;
inc0  = [5 10 20 30 40 50 60 70 80 90 100] * deg;
RAAN0 = 0.0* deg;
w0    = 0.0* deg;
M0    = 0.0;
u0    = w0 + M0;

ex0 = e0*cos(w0);
ey0 = e0*sin(w0);


%% ------------------------------------------------------------------------
% Parameters
%% ------------------------------------------------------------------------
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 4903;          % km^3/s^2 
paramsChief.muSun   = 132712;     % km^3/s^2 
paramsChief.CR      = 2; %2
paramsChief.As      = 2;           % m^2 3
paramsChief.m       = 200;           % kg 400
paramsChief.S       = 1367;          % W/m^2 1367
paramsChief.c       = 2.998e8;       % m/s
paramsChief.jd0     = juliandate(datetime(2026,1,1,0,0,0));
paramsChief.ephemModel = '421';
paramsChief.useShadow = true;


%% ------------------------------------------------------------------------
% Time span: 5 orbits
%% ------------------------------------------------------------------------
tf=365*24*60*60;

nout  = round(tf/(12*60*60));
tspan = linspace(0, tf, nout);

% Optional ephemeris precompute for compatibility with full pipeline
paramsChief.ephem  = precompute_ephemeris(tspan, paramsChief);

opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

%% ------------------------------------------------------------------------
% Propagate chief and deputy
%% ------------------------------------------------------------------------

for ii=1:length(inc0)
xc0 = [a0; ex0; ey0; inc0(ii); RAAN0; u0];

[t, xc] = ode45(@(t,x) rates_qns_total(t,x,paramsChief),  tspan, xc0, opts);

f=figure('Name',['Propagated Chief'],'Color','w');
roeLabels = {'a [km]', 'e_x', 'e_y', 'i [deg]', '\Omega [deg]','u [deg]'};

for j = 1:6
    if j~=1 && j~=2 && j~=3
    fact=180/pi;
    else
    fact=1;
    end
    subplot(3,2,j)
    plot(t./(24*60*60), xc(:,j).*fact, 'k', 'LineWidth', 1.3);
    xlabel("Time (days)")
    ylabel(roeLabels{j}, 'Interpreter','tex')
    title(roeLabels{j}, 'Interpreter','tex')
    grid on
end
saveas(f,strcat("inc_",num2str(inc0(ii)*180/pi)))
end