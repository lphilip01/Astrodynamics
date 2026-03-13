function xdot = lowThrust_MEE_dynamics(x, u, tau, auxdata)
% =========================================================================
% lowThrust_MEE_dynamics.m  (v3)
%
% MEE dynamics with J2/J3/J4 + EP thrust. SI units throughout.
% Compatible with numeric (double) and CasADi SX/MX symbolic inputs.
%
% State:   x = [p; f; g; h; k; L; w]
% Control: u = [ur; ut; uh]  (unit thrust direction in RTN)
% Param:   tau in [-50, 0]   (throttle: T_eff = T*(1 + 0.01*tau))
%
% Key:  tau = 0   -> full thrust T
%       tau = -50 -> half thrust 0.5*T
%
% No gs factor in thrust — SI: F[N]/m[kg] = a[m/s^2] directly.
% =========================================================================

mu  = auxdata.mu;
T   = auxdata.T;
Isp = auxdata.Isp;
Re  = auxdata.Re;
J2  = auxdata.J2;
J3  = auxdata.J3;
J4  = auxdata.J4;

p = x(1); f = x(2); g = x(3);
h = x(4); k = x(5); L = x(6); w = x(7);
ur = u(1); ut = u(2); uh = u(3);

% --- Auxiliary ---
q      = 1 + f*cos(L) + g*sin(L);
r      = p / q;
alpha2 = h^2 - k^2;
s2     = 1 + h^2 + k^2;

% --- ECI position ---
rX = (r/s2)*(cos(L) + alpha2*cos(L) + 2*h*k*sin(L));
rY = (r/s2)*(sin(L) - alpha2*sin(L) + 2*h*k*cos(L));
rZ = (2*r/s2)*(h*sin(L) - k*cos(L));

% --- ECI velocity ---
vX = -(1/s2)*sqrt(mu/p)*(sin(L)+alpha2*sin(L)-2*h*k*cos(L)+g-2*f*h*k+alpha2*g);
vY = -(1/s2)*sqrt(mu/p)*(-cos(L)+alpha2*cos(L)+2*h*k*sin(L)-f+2*g*h*k+alpha2*f);
vZ =  (2/s2)*sqrt(mu/p)*(h*cos(L)+k*sin(L)+f*h+g*k);

rMag = sqrt(rX^2 + rY^2 + rZ^2);

% --- RTN basis ---
ir = [rX; rY; rZ] / rMag;
hv = cross3([rX;rY;rZ],[vX;vY;vZ]);
hMag = sqrt(hv(1)^2+hv(2)^2+hv(3)^2);
ih_v = hv / hMag;
it_v = cross3(hv,[rX;rY;rZ]) / (hMag*rMag);

% --- Oblateness: geocentric latitude ---
sinphi = rZ / rMag;          % sin(geocentric latitude)
cosphi = sqrt(1 - sinphi^2);

P2  = (3*sinphi^2 - 1)/2;
P3  = (5*sinphi^3 - 3*sinphi)/2;
P4  = (35*sinphi^4 - 30*sinphi^2 + 3)/8;
dP2 = 3*sinphi;
dP3 = (15*sinphi^2 - 3)/2;
dP4 = (140*sinphi^3 - 60*sinphi)/8;

% North unit vector
en_dot_ir = ir(3);
in_u = [-en_dot_ir*ir(1); -en_dot_ir*ir(2); 1-en_dot_ir*ir(3)];
in_v = in_u / sqrt(in_u(1)^2 + in_u(2)^2 + in_u(3)^2);

sumn = (Re/r)^2*dP2*J2 + (Re/r)^3*dP3*J3 + (Re/r)^4*dP4*J4;
sumr = 3*(Re/r)^2*P2*J2 + 4*(Re/r)^3*P3*J3 + 5*(Re/r)^4*P4*J4;

dg_n = -(mu*cosphi/r^2)*sumn;
dg_r = -(mu/r^2)*sumr;

dg_ECI = dg_n*in_v - dg_r*ir;

Dg1 = ir(1)*dg_ECI(1)   + ir(2)*dg_ECI(2)   + ir(3)*dg_ECI(3);
Dg2 = it_v(1)*dg_ECI(1) + it_v(2)*dg_ECI(2) + it_v(3)*dg_ECI(3);
Dg3 = ih_v(1)*dg_ECI(1) + ih_v(2)*dg_ECI(2) + ih_v(3)*dg_ECI(3);

% --- Thrust acceleration (SI: no gs factor) ---
T_eff = T * (1 + 0.01*tau);
at    = T_eff / w;

D1 = Dg1 + at*ur;
D2 = Dg2 + at*ut;
D3 = Dg3 + at*uh;

% --- MEE EOMs ---
sq = sqrt(p/mu);

dp = (2*p/q)*sq*D2;
df =  sq*sin(L)*D1 + sq*(1/q)*((q+1)*cos(L)+f)*D2 - sq*(g/q)*(h*sin(L)-k*cos(L))*D3;
dg = -sq*cos(L)*D1 + sq*(1/q)*((q+1)*sin(L)+g)*D2 + sq*(f/q)*(h*sin(L)-k*cos(L))*D3;
dh = sq*(s2*cos(L)/(2*q))*D3;
dk = sq*(s2*sin(L)/(2*q))*D3;
dL = sq*(1/q)*(h*sin(L)-k*cos(L))*D3 + sqrt(mu*p)*(q/p)^2;
g0 = 9.80665;   % [m/s^2]
dw = -(T_eff / (Isp * g0));

xdot = [dp; df; dg; dh; dk; dL; dw];
end

function c = cross3(a, b)
c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)];
end
