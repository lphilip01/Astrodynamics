function sol = solve_qns_free_time_ocp(ref, deputyInit, params, prob)
% solve_qns_free_time_ocp
%
% Single-deputy, free-final-time Bolza optimal control problem in QNS
% elements using CasADi + IPOPT and the same RK4 single-step architecture
% used in the science-hold OCP.
%
% Problem:
%   min_{Tf, v(.)}
%       wf*Tf ...
%     + wm*(m(0)-m(Tf)) ...
%     + epsU * integral_0^Tf ||v||^2 dt ...
%     + wterm * ||deltaAlpha(Tf) - deltaAlphaTarget||_W^2
%
% Subject to:
%   xdot = f_QNS(x, v, rSun, rMoon)
%   x(0) = x0
%   ||v|| <= 1
%   m >= mdry
%   m(k+1) <= m(k)
%   ||delta r|| >= dMin
%   TfMin <= Tf <= TfMax
%
% State:
%   x = [a; ex; ey; inc; RAAN; u; m]
%
% Control:
%   v = [vR; vT; vN], with ||v|| <= 1
%
% Relative terminal element vector:
%   deltaAlpha = [delta a; delta lambda; delta ex; delta ey; delta ix; delta iy]
%
% Inputs
% ------
% ref.tGrid    : (NtRef x 1) physical time grid [s]
% ref.chiefHist: (NtRef x 6) chief QNS history [a ex ey i RAAN u]
% ref.rSun     : (NtRef x 3) Sun position history in ECI [km]
% ref.rMoon    : (NtRef x 3) Moon position history in ECI [km]
%
% deputyInit   : (7 x 1) initial deputy state [a ex ey i RAAN u m]
% params       : dynamics/propulsion parameter struct
%
% Required prob fields
% --------------------
% prob.N                  : number of control intervals
% prob.TfMin              : lower bound on final time [s]
% prob.TfMax              : upper bound on final time [s]
% prob.TfGuess            : initial guess for final time [s]
% prob.massDry_kg         : dry mass lower bound [kg]
% prob.dMin_km            : minimum separation from chief [km]
% prob.deltaAlphaTarget   : (6 x 1) desired terminal relative QNS vector
% prob.Wterm              : (6 x 6) weight matrix or (6 x 1) diagonal weights
% prob.wf                 : weight on final time
% prob.wm                 : weight on propellant use
% prob.epsU               : weight on integral ||v||^2 regularization
% prob.wterm              : scalar weight on terminal penalty
%
% Optional prob fields
% --------------------
% prob.t0                 : initial physical time [s], default ref.tGrid(1)
% prob.maxOuterIter       : ephemeris/chief resampling iterations, default 3
% prob.outerTol           : relative Tf convergence tolerance, default 1e-3
% prob.interpMethod       : interp1 method, default 'pchip'
% prob.unwrapChiefAngles  : unwrap chief RAAN/u before interpolation, default true
% prob.ipoptMaxIter       : default 600
% prob.ipoptTol           : default 1e-6
% prob.ipoptPrintLevel    : default 5
%
% Notes
% -----
% 1) The NLP is built in normalized time tau in [0,1], with h = Tf/N.
% 2) Chief and ephemerides are sampled outside the NLP on the normalized
%    grid using the current Tf reference, then held fixed during each NLP
%    solve. A short outer loop refreshes these samples after Tf updates.
% 3) For angle interpolation, chief RAAN and chief u are unwrapped by
%    default. The input chief history should still be physically continuous.
%
% A few practical Wterm design choices:
%   - Identity: W = eye(6)
%   - Scale-normalized: W = diag(1./sigma.^2)
%   - Priority-weighted: W = diag([wa wl wex wey wix wiy])

import casadi.*

validate_inputs(ref, deputyInit, prob);
prob = apply_defaults(prob, ref);
Wterm = make_weight_matrix(prob.Wterm);

tau = linspace(0, 1, prob.N + 1).';
F_rk4 = compile_variable_step_rk4(params);

TfRef = prob.TfGuess;
Xguess = [];
Uguess = [];

outerHist = repmat(struct( ...
    'iter', [], ...
    'TfReference_s', [], ...
    'TfSolved_s', [], ...
    'objective', [], ...
    'success', []), prob.maxOuterIter, 1);

for outerIter = 1:prob.maxOuterIter
    sample = sample_reference_on_tau(ref, tau, prob.t0, TfRef, prob);

    if isempty(Xguess) || isempty(Uguess)
        [Xguess, Uguess] = build_zero_control_guess(deputyInit, params, sample, TfRef);
    end

    phase = build_and_solve_free_time_phase( ...
        F_rk4, sample, deputyInit, params, prob, Wterm, Xguess, Uguess, TfRef);

    outerHist(outerIter).iter = outerIter;
    outerHist(outerIter).TfReference_s = TfRef;
    outerHist(outerIter).TfSolved_s = phase.Tf_s;
    outerHist(outerIter).objective = phase.objective;
    outerHist(outerIter).success = phase.success;

    Xguess = phase.X.';
    Uguess = phase.U.';

    TfErr = abs(phase.Tf_s - TfRef) / max(1, abs(TfRef));
    TfRef = phase.Tf_s;

    if TfErr <= prob.outerTol
        break;
    end
end

outerHist = outerHist(~arrayfun(@isempty, {outerHist.iter}));

sol = phase;
sol.outerHistory = outerHist;
sol.outerConverged = ~isempty(outerHist) & ...
    abs(outerHist(end).TfSolved_s - outerHist(end).TfReference_s) / ...
    max(1, abs(outerHist(end).TfReference_s)) <= prob.outerTol;
sol.problem = prob;
sol.Wterm = Wterm;

end

% =========================================================================
function prob = apply_defaults(prob, ref)
if ~isfield(prob, 't0') || isempty(prob.t0), prob.t0 = ref.tGrid(1); end
if ~isfield(prob, 'maxOuterIter') || isempty(prob.maxOuterIter), prob.maxOuterIter = 3; end
if ~isfield(prob, 'outerTol') || isempty(prob.outerTol), prob.outerTol = 1e-3; end
if ~isfield(prob, 'interpMethod') || isempty(prob.interpMethod), prob.interpMethod = 'pchip'; end
if ~isfield(prob, 'unwrapChiefAngles') || isempty(prob.unwrapChiefAngles), prob.unwrapChiefAngles = true; end
if ~isfield(prob, 'ipoptMaxIter') || isempty(prob.ipoptMaxIter), prob.ipoptMaxIter = 600; end
if ~isfield(prob, 'ipoptTol') || isempty(prob.ipoptTol), prob.ipoptTol = 1e-6; end
if ~isfield(prob, 'ipoptPrintLevel') || isempty(prob.ipoptPrintLevel), prob.ipoptPrintLevel = 5; end
end

% =========================================================================
function validate_inputs(ref, deputyInit, prob)
requiredRef = {'tGrid', 'chiefHist', 'rSun', 'rMoon'};
for k = 1:numel(requiredRef)
    if ~isfield(ref, requiredRef{k})
        error('Missing ref.%s', requiredRef{k});
    end
end

requiredProb = {'N', 'TfMin', 'TfMax', 'TfGuess', 'massDry_kg', 'dMin_km', ...
                'deltaAlphaTarget', 'Wterm', 'wf', 'wm', 'epsU', 'wterm'};
for k = 1:numel(requiredProb)
    if ~isfield(prob, requiredProb{k})
        error('Missing prob.%s', requiredProb{k});
    end
end

if numel(deputyInit) ~= 7
    error('deputyInit must be a 7x1 state vector.');
end

if size(ref.chiefHist,2) ~= 6
    error('ref.chiefHist must have 6 columns: [a ex ey i RAAN u].');
end

if size(ref.rSun,2) ~= 3 || size(ref.rMoon,2) ~= 3
    error('ref.rSun and ref.rMoon must be NtRef x 3.');
end

NtRef = numel(ref.tGrid);
if size(ref.chiefHist,1) ~= NtRef || size(ref.rSun,1) ~= NtRef || size(ref.rMoon,1) ~= NtRef
    error('ref histories must all share the same first dimension as ref.tGrid.');
end

if prob.TfMin <= 0 || prob.TfMax <= 0 || prob.TfGuess <= 0
    error('TfMin, TfMax, and TfGuess must be positive.');
end

if ~(prob.TfMin <= prob.TfGuess && prob.TfGuess <= prob.TfMax)
    error('prob.TfGuess must satisfy TfMin <= TfGuess <= TfMax.');
end
end

% =========================================================================
function W = make_weight_matrix(Win)
if isvector(Win)
    if numel(Win) ~= 6
        error('If prob.Wterm is a vector, it must have 6 entries.');
    end
    W = diag(Win(:));
else
    if ~isequal(size(Win), [6 6])
        error('prob.Wterm must be 6x6 or a 6x1 diagonal weight vector.');
    end
    W = Win;
end

W = 0.5 * (W + W.');
end

% =========================================================================
function F_rk4 = compile_variable_step_rk4(params)
import casadi.*

x_cas     = MX.sym('x', 7);
u_cas     = MX.sym('u', 3);
h_cas     = MX.sym('h', 1);
rSun_cas  = MX.sym('rSun', 3);
rMoon_cas = MX.sym('rMoon', 3);

f1 = qns_free_time_dynamics_casadi(x_cas,                 u_cas, params, rSun_cas, rMoon_cas);
f2 = qns_free_time_dynamics_casadi(x_cas + 0.5*h_cas*f1, u_cas, params, rSun_cas, rMoon_cas);
f3 = qns_free_time_dynamics_casadi(x_cas + 0.5*h_cas*f2, u_cas, params, rSun_cas, rMoon_cas);
f4 = qns_free_time_dynamics_casadi(x_cas + h_cas*f3,     u_cas, params, rSun_cas, rMoon_cas);

xnext_expr = x_cas + (h_cas/6) * (f1 + 2*f2 + 2*f3 + f4);

F_rk4 = casadi.Function('F_rk4_free_tf', ...
    {x_cas, u_cas, h_cas, rSun_cas, rMoon_cas}, ...
    {xnext_expr}, ...
    {'x', 'u', 'h', 'rSun', 'rMoon'}, {'xnext'});
end

% =========================================================================
function sample = sample_reference_on_tau(ref, tau, t0, TfRef, prob)
tNodes = t0 + TfRef * tau(:);

if tNodes(1) < ref.tGrid(1) || tNodes(end) > ref.tGrid(end)
    error(['Reference sampling interval [%.3f, %.3f] exceeds the available ' ...
           'reference time span [%.3f, %.3f].'], ...
           tNodes(1), tNodes(end), ref.tGrid(1), ref.tGrid(end));
end

chiefHist = ref.chiefHist;
if prob.unwrapChiefAngles
    chiefHist(:,5) = unwrap(chiefHist(:,5));
    chiefHist(:,6) = unwrap(chiefHist(:,6));
end

sample = struct();
sample.tau = tau(:);
sample.t = tNodes(:);
sample.TfReference_s = TfRef;
sample.chief = interp1(ref.tGrid(:), chiefHist, tNodes, prob.interpMethod);
sample.rSun = interp1(ref.tGrid(:), ref.rSun, tNodes, prob.interpMethod);
sample.rMoon = interp1(ref.tGrid(:), ref.rMoon, tNodes, prob.interpMethod);
end

% =========================================================================
function [Xguess, Uguess] = build_zero_control_guess(deputyInit, params, sample, TfRef)
Nt = numel(sample.t);
N = Nt - 1;
h = TfRef / N;

Xguess = zeros(7, Nt);
Uguess = zeros(3, N);
Xguess(:,1) = deputyInit(:);

for k = 1:N
    rSm = 0.5 * (sample.rSun(k,:).' + sample.rSun(k+1,:).');
    rMm = 0.5 * (sample.rMoon(k,:).' + sample.rMoon(k+1,:).');
    xk = Xguess(:,k);

    f1 = qns_free_time_dynamics_casadi(xk,             zeros(3,1), params, rSm, rMm);
    f2 = qns_free_time_dynamics_casadi(xk + 0.5*h*f1, zeros(3,1), params, rSm, rMm);
    f3 = qns_free_time_dynamics_casadi(xk + 0.5*h*f2, zeros(3,1), params, rSm, rMm);
    f4 = qns_free_time_dynamics_casadi(xk + h*f3,     zeros(3,1), params, rSm, rMm);

    Xguess(:,k+1) = xk + (h/6) * (f1 + 2*f2 + 2*f3 + f4);
end
end

% =========================================================================
function sol = build_and_solve_free_time_phase(F_rk4, sample, deputyInit, params, prob, Wterm, Xinit, Uinit, TfInit)
import casadi.*

Nt = numel(sample.t);
N = Nt - 1;

opti = casadi.Opti();

X = opti.variable(7, Nt);
U = opti.variable(3, N);
Tf = opti.variable();

opti.subject_to(X(:,1) == deputyInit(:));
opti.subject_to(Tf >= prob.TfMin);
opti.subject_to(Tf <= prob.TfMax);

h = Tf / N;

for k = 1:N
    rSm = 0.5 * (sample.rSun(k,:).' + sample.rSun(k+1,:).');
    rMm = 0.5 * (sample.rMoon(k,:).' + sample.rMoon(k+1,:).');
    xnext = F_rk4(X(:,k), U(:,k), h, rSm, rMm);
    opti.subject_to(X(:,k+1) == xnext);
end

for k = 1:N
    opti.subject_to(sumsqr(U(:,k)) <= 1.0);
end

opti.subject_to(vec(X(7,:).') >= prob.massDry_kg);
opti.subject_to(X(7,2:end) <= X(7,1:end-1));

for k = 1:Nt
    chiefk = sample.chief(k,:).';
    deltaAlpha_k = relative_qns_elements(X(1:6,k), chiefk);
    deltaR_k = relative_rtn_from_delta_alpha(deltaAlpha_k, chiefk);
    opti.subject_to(sumsqr(deltaR_k) >= prob.dMin_km^2);
end

deltaAlpha_f = relative_qns_elements(X(1:6,end), sample.chief(end,:).');
deltaAlphaErr = deltaAlpha_f - prob.deltaAlphaTarget(:);

Jtime = prob.wf * Tf;
Jmass = prob.wm * (deputyInit(7) - X(7,end));

Jctrl = 0;
for k = 1:N
    Jctrl = Jctrl + prob.epsU * h * sumsqr(U(:,k));
end

Jterm = prob.wterm * (deltaAlphaErr.' * Wterm * deltaAlphaErr);
J = Jtime + Jmass + Jctrl + Jterm;

opti.minimize(J);

opti.set_initial(X, Xinit);
opti.set_initial(U, Uinit);
opti.set_initial(Tf, TfInit);

opts = struct();
opts.ipopt.max_iter = prob.ipoptMaxIter;
opts.ipopt.tol = prob.ipoptTol;
opts.ipopt.print_level = prob.ipoptPrintLevel;
opts.ipopt.mu_strategy = 'adaptive';
opts.ipopt.nlp_scaling_method = 'gradient-based';
opts.ipopt.linear_solver = 'mumps';

opti.solver('ipopt', opts);

try
    S = opti.solve();
    success = true;
catch ME
    warning(ME.identifier,'Free-time QNS OCP failed: %s', ME.message);
    S = opti.debug;
    success = false;
end

sol = struct();
sol.success = success;
sol.tau = sample.tau;
sol.t = sample.t;
sol.referenceSampleTf_s = sample.TfReference_s;
sol.Tf_s = S.value(Tf);
sol.h_s = sol.Tf_s / N;
sol.X = S.value(X).';
sol.U = S.value(U).';
sol.chief = sample.chief;
sol.rSun = sample.rSun;
sol.rMoon = sample.rMoon;

sol.objective = S.value(J);
sol.Jtime = S.value(Jtime);
sol.Jmass = S.value(Jmass);
sol.Jctrl = S.value(Jctrl);
sol.Jterm = S.value(Jterm);

sol.massInitial_kg = deputyInit(7);
sol.massFinal_kg = sol.X(end,7);
sol.massUsed_kg = sol.massInitial_kg - sol.massFinal_kg;

[deltaAlphaHist, deltaRHist] = postprocess_relative_geometry(sol.X, sol.chief);
sol.deltaAlpha = deltaAlphaHist;
sol.deltaR_km = deltaRHist;
sol.rangeToChief_km = sqrt(sum(deltaRHist.^2, 2));

sol.deltaAlphaTarget = prob.deltaAlphaTarget(:).';
sol.deltaAlphaFinal = deltaAlphaHist(end,:);
sol.deltaAlphaError = sol.deltaAlphaFinal - sol.deltaAlphaTarget;
sol.terminalPenaltyRaw = sol.deltaAlphaError * Wterm * sol.deltaAlphaError.';

sol.thrustNorm = sqrt(sum(sol.U.^2, 2));
sol.dvApprox_mps = approximate_delta_v(sol.X(:,7), sol.thrustNorm, sol.h_s, params);
end

% =========================================================================
function [deltaAlphaHist, deltaRHist] = postprocess_relative_geometry(X, chief)
Nt = size(X,1);
deltaAlphaHist = zeros(Nt, 6);
deltaRHist = zeros(Nt, 3);

for k = 1:Nt
    chiefk = chief(k,:).';
    xk = X(k,1:6).';
    deltaAlpha_k = full(relative_qns_elements(xk, chiefk));
    deltaR_k = full(relative_rtn_from_delta_alpha(deltaAlpha_k, chiefk));

    deltaAlphaHist(k,:) = deltaAlpha_k(:).';
    deltaRHist(k,:) = deltaR_k(:).';
end
end

% =========================================================================
function dvApprox_mps = approximate_delta_v(mNodes, thrustNorm, h, params)
if isempty(thrustNorm)
    dvApprox_mps = 0;
    return;
end

mMid = 0.5 * (mNodes(1:end-1) + mNodes(2:end));
aMid = (params.T ./ mMid) .* thrustNorm;
dvApprox_mps = sum(aMid) * h;
end

% =========================================================================
function deltaAlpha = relative_qns_elements(xDeputy, xChief)
ad = xDeputy(1);
exd = xDeputy(2);
eyd = xDeputy(3);
id = xDeputy(4);
RAANd = xDeputy(5);
ud = xDeputy(6);

ac = xChief(1);
exc = xChief(2);
eyc = xChief(3);
ic = xChief(4);
RAANc = xChief(5);
uc = xChief(6);

dRA = RAANd - RAANc;

deltaAlpha = [ ...
    (ad - ac) / ac; ...
    (ud - uc) + dRA * cos(ic); ...
    exd - exc; ...
    eyd - eyc; ...
    id - ic; ...
    dRA * sin(ic)];
end

% =========================================================================
function deltaR = relative_rtn_from_delta_alpha(deltaAlpha, chiefState)
ac = chiefState(1);
uc = chiefState(6);

deltaA = deltaAlpha(1);
deltaLambda = deltaAlpha(2);
deltaEx = deltaAlpha(3);
deltaEy = deltaAlpha(4);
deltaIx = deltaAlpha(5);
deltaIy = deltaAlpha(6);

dR = ac * (deltaA - cos(uc) * deltaEx - sin(uc) * deltaEy);
dT = ac * (deltaLambda + 2*sin(uc) * deltaEx - 2*cos(uc) * deltaEy);
dN = ac * (sin(uc) * deltaIx - cos(uc) * deltaIy);

deltaR = [dR; dT; dN];
end

% =========================================================================
function xdot = qns_free_time_dynamics_casadi(x, vctrl, params, rSun, rMoon)
% Dedicated free-time QNS dynamics for the single-deputy retargeting OCP.

mu     = params.mu;
RE     = params.RE;
J2     = params.J2;
muMoon = params.muMoon;
muSun  = params.muSun;
CR     = params.CR;
As     = params.As;
S      = params.S;
c      = params.c;
T      = params.T;
Isp    = params.Isp;
g0     = 9.80665;

a    = x(1);
inc  = x(4);
RAAN = x(5);
u    = x(6);
m    = x(7);

vR = vctrl(1);
vT = vctrl(2);
vN = vctrl(3);

n = sqrt(mu / a^3);
r = a;

cO = cos(RAAN); sO = sin(RAAN);
ci = cos(inc);  si = sin(inc);
cu = cos(u);    su = sin(u);

Rhat = [ cO*cu - sO*su*ci;
         sO*cu + cO*su*ci;
         su*si ];

That = [ -cO*su - sO*cu*ci;
         -sO*su + cO*cu*ci;
          cu*si ];

Nhat = [ -sO*si;
          cO*si;
         -ci ];

rSat = r * Rhat;

facJ2 = -(3/2) * J2 * mu * RE^2 / a^4;
AR_J2 = facJ2 * (1 - 3*si^2*su^2);
AT_J2 = facJ2 * (si^2 * sin(2*u));
AN_J2 = facJ2 * (sin(2*inc) * su);

if (As == 0) || (CR == 0) || (S == 0)
    AR_SRP = 0;
    AT_SRP = 0;
    AN_SRP = 0;
else
    sHat = rSun / sqrt(sum(rSun.^2));
    pSR = (S/c) * CR * As / m / 1000;
    AR_SRP = -pSR * dot(sHat, Rhat);
    AT_SRP = -pSR * dot(sHat, That);
    AN_SRP = -pSR * dot(sHat, Nhat);
end

if muMoon == 0
    AR_Moon = 0;
    AT_Moon = 0;
    AN_Moon = 0;
else
    rhoMoon = rMoon - rSat;
    aMoonECI = muMoon * ( ...
        rhoMoon / (sqrt(sum(rhoMoon.^2))^3) - ...
        rMoon  / (sqrt(sum(rMoon.^2))^3) );
    AR_Moon = dot(aMoonECI, Rhat);
    AT_Moon = dot(aMoonECI, That);
    AN_Moon = dot(aMoonECI, Nhat);
end

if muSun == 0
    AR_Sun = 0;
    AT_Sun = 0;
    AN_Sun = 0;
else
    rhoSun = rSun - rSat;
    aSunECI = muSun * ( ...
        rhoSun / (sqrt(sum(rhoSun.^2))^3) - ...
        rSun  / (sqrt(sum(rSun.^2))^3) );
    AR_Sun = dot(aSunECI, Rhat);
    AT_Sun = dot(aSunECI, That);
    AN_Sun = dot(aSunECI, Nhat);
end

athrust = (T / m) / 1000;
AR_th = athrust * vR;
AT_th = athrust * vT;
AN_th = athrust * vN;

AR = AR_J2 + AR_SRP + AR_Moon + AR_Sun + AR_th;
AT = AT_J2 + AT_SRP + AT_Moon + AT_Sun + AT_th;
AN = AN_J2 + AN_SRP + AN_Moon + AN_Sun + AN_th;

sinISafe = sin(inc) + 1e-8;

a_dot    = (2/n) * AT;
u_dot    = n - (2/(n*a))*AR - (cos(inc)/(n*a*sinISafe)) * sin(u) * AN;
ex_dot   = (1/(n*a)) * ( sin(u)*AR + 2*cos(u)*AT );
ey_dot   = (1/(n*a)) * ( -cos(u)*AR + 2*sin(u)*AT );
inc_dot  = (1/(n*a)) * cos(u) * AN;
RAAN_dot = (1/(n*a*sinISafe)) * sin(u) * AN;

epsThrottle = 1e-10;
normSq = vR^2 + vT^2 + vN^2;
throttle = sqrt(normSq + epsThrottle^2) - epsThrottle;
m_dot = -(T * throttle) / (Isp * g0);

xdot = [a_dot; ex_dot; ey_dot; inc_dot; RAAN_dot; u_dot; m_dot];
end
