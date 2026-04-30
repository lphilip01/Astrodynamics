function [r_eci, v_eci] = qns_to_eci(x, mu)
% x = [a; ex; ey; inc; RAAN; u], with u = omega + M

a    = x(1);
ex   = x(2);
ey   = x(3);
inc  = x(4);
RAAN = x(5);
u    = x(6);

e = hypot(ex,ey);
omega = 0;
if e > 1e-12
    omega = atan2(ey,ex);
end

M = mod(u - omega, 2*pi);

% Kepler solve
E = M;
for k = 1:20
    f = E - e*sin(E) - M;
    fp = 1 - e*cos(E);
    dE = -f/fp;
    E = E + dE;
    if abs(dE) < 1e-13, break; end
end

nu = 2*atan2(sqrt(1+e)*sin(E/2), sqrt(1-e)*cos(E/2));
p  = a*(1-e^2);

r_pf = [p*cos(nu)/(1+e*cos(nu));
        p*sin(nu)/(1+e*cos(nu));
        0];

v_pf = sqrt(mu/p) * [-sin(nu);
                      e+cos(nu);
                      0];

cO = cos(RAAN); sO = sin(RAAN);
ci = cos(inc);  si = sin(inc);
co = cos(omega); so = sin(omega);

R3_O = [cO -sO 0; sO cO 0; 0 0 1];
R1_i = [1 0 0; 0 ci -si; 0 si ci];
R3_o = [co -so 0; so co 0; 0 0 1];

Q = R3_O*R1_i*R3_o;

r_eci = Q*r_pf;
v_eci = Q*v_pf;
end