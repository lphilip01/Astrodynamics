function [RA_deg, DEC_deg, target_unit] = generate_equal_area_sky(N)
% Equal-area sky sampling using Fibonacci sphere

golden = (1 + sqrt(5))/2;

k = (0:N-1)';

z = 1 - 2*(k+0.5)/N;
theta = 2*pi*k/golden;

r = sqrt(1 - z.^2);

x = r .* cos(theta);
y = r .* sin(theta);

target_unit = [x y z];

DEC_deg = asind(z);
RA_deg  = mod(rad2deg(atan2(y,x)),360);
end