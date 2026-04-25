function sol = solve_science_hold_ocp(tGrid, chiefInput, deputyInitCell, params, target, prob)
% solve_science_hold_ocp
%
% Fixed-duration science-hold OCP with two supported modes:
%   1) Legacy fixed-chief mode:
%        chiefInput = Nt x 6 chief reference trajectory
%   2) Controlled-chief mode:
%        chiefInput = 7x1 chief initial state
%        chiefInput = struct('x0', chiefX0, 'guess', chiefGuessHist)
%        chiefInput = Nt x 6/7 history with prob.controlChief = true
%
% The controlled-chief formulation adds:
%   - distributed control across chief + collectors
%   - time-averaged RMS OPD constraints with one slack per collector
%   - optional thrust-cone constraints in RTN
%
% Inputs:
%   tGrid          : (Nt x 1) time grid [s]
%   chiefInput     : chief history or chief initial condition
%   deputyInitCell : 1xNc cell array, each 7x1:
%                    [a; ex; ey; inc; RAAN; u; m]
%   params         : struct with dynamics/propulsion parameters
%   target         : struct with target definition
%   prob           : struct with OCP settings
%
% Required prob fields:
%   prob.Dmax_m
%   prob.rhoMin_km
%   prob.dPairMin_km
%   prob.massDry_kg
%
% Optional controlled-chief fields:
%   prob.controlChief         logical, when chiefInput is a history and the
%                             chief should become an optimization state
%   prob.chiefMass0_kg        chief initial mass if chiefInput is 6-state
%   prob.massDryChief_kg      chief dry mass if different from collectors
%   prob.thetaMax_rad / deg   scalar, Nopt-vector, or Nc-vector
%   prob.thetaMaxChief_rad    chief-specific cone half-angle
%   prob.thetaMaxCollector_rad / deg
%   prob.thrustAxisRTN        RTN thrust-axis data
%
% Accepted prob.thrustAxisRTN formats:
%   - 3x1                   common constant axis
%   - 3xNopt or Noptx3      constant per optimized spacecraft
%   - 3xNc or Ncx3          constant per collector (chief given separately)
%   - 3xNint or Nintx3      common time-varying axis over control intervals
%   - cell array            per-spacecraft axis specs; each cell may be any
%                           of the numeric formats above
%
% Notes:
%   - The solver keeps the previous two-phase structure.
%   - Legacy fields sol.chief, sol.X, sol.U, sol.RTN, and sol.OPD_km are
%     preserved so existing post-processing continues to work.

import casadi.*

if nargin < 6
    error('solve_science_hold_ocp requires tGrid, chiefInput, deputyInitCell, params, target, and prob.');
end

tGrid = tGrid(:);
Nt = numel(tGrid);
if Nt < 2
    error('tGrid must contain at least two nodes.');
end

dt = diff(tGrid);
if max(abs(dt - dt(1))) > 1e-9
    error('tGrid must be uniform for this version.');
end
h = dt(1);

requiredProbFields = {'Dmax_m', 'rhoMin_km', 'dPairMin_km', 'massDry_kg'};
for k = 1:numel(requiredProbFields)
    if ~isfield(prob, requiredProbFields{k})
        error('Missing prob.%s.', requiredProbFields{k});
    end
end

if ~isfield(params, 'ephem') || ~isfield(params.ephem, 'rSun') || ~isfield(params.ephem, 'rMoon')
    error('params.ephem.rSun and params.ephem.rMoon are required.');
end

rSunTab = params.ephem.rSun;
rMoonTab = params.ephem.rMoon;
if size(rSunTab, 1) ~= Nt || size(rMoonTab, 1) ~= Nt || size(rSunTab, 2) ~= 3 || size(rMoonTab, 2) ~= 3
    error('params.ephem.rSun and params.ephem.rMoon must be Nt x 3 over tGrid.');
end

% ---------------- Defaults ----------------
if ~isfield(prob, 'phase1ConstraintStride'), prob.phase1ConstraintStride = 5; end
if ~isfield(prob, 'phase1_wSlack'),          prob.phase1_wSlack = 1; end
if ~isfield(prob, 'phase1_wSmooth'),         prob.phase1_wSmooth = 1e-4; end

if ~isfield(prob, 'phase2_wSlack')
    if isfield(prob, 'wSlack')
        prob.phase2_wSlack = prob.wSlack;
    else
        prob.phase2_wSlack = 1e3;
    end
end
if ~isfield(prob, 'phase2_wFuel')
    if isfield(prob, 'wFuel')
        prob.phase2_wFuel = prob.wFuel;
    else
        prob.phase2_wFuel = 1;
    end
end
if ~isfield(prob, 'phase2_wControl')
    if isfield(prob, 'wControl')
        prob.phase2_wControl = prob.wControl;
    else
        prob.phase2_wControl = 1e-5;
    end
end
if ~isfield(prob, 'phase2_wSmooth')
    if isfield(prob, 'wSmooth')
        prob.phase2_wSmooth = prob.wSmooth;
    else
        prob.phase2_wSmooth = 1e-4;
    end
end

setup = local_prepare_setup(tGrid, chiefInput, deputyInitCell, params, prob);

% =========================================================================
% Compile spacecraft-specific CasADi RK4 functions
% =========================================================================
fprintf('  Compiling RK4 CasADi Functions...');
F_rk4_cell = local_compile_rk4_functions(h, setup.spacecraftParams);
fprintf(' done.\n');

% =========================================================================
% Dynamically consistent initial guess
% =========================================================================
[XguessCell, UguessCell] = local_build_initial_guess(tGrid, rSunTab, rMoonTab, setup);

% =========================================================================
% Phase 1 solve
% =========================================================================
phase1 = build_and_solve_phase( ...
    'phase1', tGrid, target, prob, setup, F_rk4_cell, ...
    XguessCell, UguessCell, zeros(setup.Nc, 1), ...
    prob.phase1ConstraintStride, ...
    prob.phase1_wSlack, 0, 0, prob.phase1_wSmooth);

% =========================================================================
% Phase 2 solve
% =========================================================================
phase2 = build_and_solve_phase( ...
    'phase2', tGrid, target, prob, setup, F_rk4_cell, ...
    phase1.Xwarm, phase1.Uwarm, phase1.Xi, ...
    1, ...
    prob.phase2_wSlack, prob.phase2_wFuel, prob.phase2_wControl, prob.phase2_wSmooth);

% =========================================================================
% Final packaging
% =========================================================================
sol = local_finalize_solution(phase2, setup, tGrid, target, prob);
sol.phase1 = phase1;
sol.phase2 = phase2;

end

% =========================================================================
function phase = build_and_solve_phase(phaseName, tGrid, target, prob, setup, F_rk4_cell, ...
                                       XinitCell, UinitCell, XiInit, ...
                                       constraintStride, wSlack, wFuel, wControl, wSmooth)

import casadi.*

Nt = numel(tGrid);
Nint = Nt - 1;
Nc = setup.Nc;
offset = double(setup.controlChief);
Dtol_km = (prob.Dmax_m / 1000) / 2;
Tspan = tGrid(end) - tGrid(1);

rSunTab = setup.ephem.rSun;
rMoonTab = setup.ephem.rMoon;

sampleIdx = local_sample_indices(Nt, constraintStride);
tSample = tGrid(sampleIdx);
wSample = local_trapz_weights(tSample);

opti = casadi.Opti();

X = cell(1, setup.Nopt);
U = cell(1, setup.Nopt);
for s = 1:setup.Nopt
    X{s} = opti.variable(7, Nt);
    U{s} = opti.variable(3, Nint);
end

Xi = opti.variable(Nc, 1);
opti.subject_to(Xi >= 0);

% -------------------------------------------------------------------------
% Initial conditions
% -------------------------------------------------------------------------
for s = 1:setup.Nopt
    opti.subject_to(X{s}(:,1) == setup.initStateCell{s});
end

% -------------------------------------------------------------------------
% Dynamics constraints
% -------------------------------------------------------------------------
for k = 1:Nint
    rSm = 0.5 * (rSunTab(k,:).' + rSunTab(k+1,:).');
    rMm = 0.5 * (rMoonTab(k,:).' + rMoonTab(k+1,:).');

    for s = 1:setup.Nopt
        xnext = F_rk4_cell{s}(X{s}(:,k), U{s}(:,k), rSm, rMm);
        opti.subject_to(X{s}(:,k+1) == xnext);
    end
end

% -------------------------------------------------------------------------
% Control, mass, and thrust-cone constraints
% -------------------------------------------------------------------------
for s = 1:setup.Nopt
    for k = 1:Nint
        uk = U{s}(:,k);
        opti.subject_to(sumsqr(uk) <= 1.0);

        if setup.coneEnabled(s)
            bHat = local_get_thrust_axis_rtn(prob, s, k, setup.Nopt, setup.Nc, setup.controlChief, Nint, Nt);
            uMag = local_smooth_norm(uk);
            opti.subject_to(sum1(uk .* bHat) >= cos(setup.thetaMaxRad(s)) * uMag);
        end
    end

    opti.subject_to(vec(X{s}(1,:).') >= setup.aMinKm(s));
    if isfinite(setup.aMaxKm(s))
        opti.subject_to(vec(X{s}(1,:).') <= setup.aMaxKm(s));
    end
    opti.subject_to(vec(X{s}(7,:).') >= setup.massDryKg(s));
    opti.subject_to(X{s}(7,2:end) <= X{s}(7,1:end-1));
end

% -------------------------------------------------------------------------
% Science and formation path constraints
% -------------------------------------------------------------------------
opdRelSqAccum = MX.zeros(Nc, 1);

for ii = 1:numel(sampleIdx)
    k = sampleIdx(ii);

    if setup.controlChief
        xChief = X{1}(1:6, k);
    else
        xChief = setup.chiefHist(k,:).';
    end

    sRTN = local_source_rtn_from_target(target, xChief);

    opd_k = MX.zeros(Nc, 1);
    drMat = MX.zeros(3, Nc);

    for j = 1:Nc
        xDep = X{j + offset}(1:6, k);
        dr = local_relative_rtn_from_qns(xChief, xDep);

        drMat(:,j) = dr;
        opd_k(j) = sum1(dr .* sRTN);
    end

    opdMean = sum1(opd_k) / Nc;

    for j = 1:Nc
        opdRel = opd_k(j) - opdMean;
        opdRelSqAccum(j) = opdRelSqAccum(j) + wSample(ii) * opdRel^2;

        rj2 = sumsqr(drMat(:,j));
        opti.subject_to(rj2 >= prob.rhoMin_km^2);
        if isfield(prob, 'rhoMax_km') && ~isempty(prob.rhoMax_km)
            opti.subject_to(rj2 <= prob.rhoMax_km^2);
        end
    end

    for i = 1:Nc-1
        for j = i+1:Nc
            dij2 = sumsqr(drMat(:,i) - drMat(:,j));
            opti.subject_to(dij2 >= prob.dPairMin_km^2);
        end
    end
end

for j = 1:Nc
    opti.subject_to(opdRelSqAccum(j) <= Tspan * (Dtol_km^2 + Xi(j)));
end

% -------------------------------------------------------------------------
% Objective
% -------------------------------------------------------------------------
Jfuel = 0;
Jctrl = 0;
Jsmooth = 0;

for s = 1:setup.Nopt
    m0 = setup.initStateCell{s}(7);
    mf = X{s}(7,end);
    Jfuel = Jfuel + (m0 - mf);

    for k = 1:Nint
        Jctrl = Jctrl + wControl * (tGrid(k+1) - tGrid(k)) * sumsqr(U{s}(:,k));
    end

    for k = 1:Nint-1
        du = U{s}(:,k+1) - U{s}(:,k);
        Jsmooth = Jsmooth + (wSmooth / max(tGrid(k+1) - tGrid(k), eps)) * sumsqr(du);
    end
end

Jslack = wSlack * sumsqr(Xi);
J = wFuel * Jfuel + Jctrl + Jsmooth + Jslack;
opti.minimize(J);

% -------------------------------------------------------------------------
% Initial guess
% -------------------------------------------------------------------------
for s = 1:setup.Nopt
    opti.set_initial(X{s}, XinitCell{s});
    opti.set_initial(U{s}, UinitCell{s});
end
opti.set_initial(Xi, XiInit(:));

% -------------------------------------------------------------------------
% Solver options
% -------------------------------------------------------------------------
opts = struct();
opts.ipopt.max_iter = 500;
opts.ipopt.tol = 1e-5;
opts.ipopt.print_level = 5;
opts.ipopt.mu_strategy = 'adaptive';
opts.ipopt.nlp_scaling_method = 'gradient-based';
opts.ipopt.linear_solver = 'mumps';

opti.solver('ipopt', opts);

% -------------------------------------------------------------------------
% Solve
% -------------------------------------------------------------------------
try
    S = opti.solve();
    success = true;
catch ME
    warning('%s failed: %s', phaseName, ME.message);
    S = opti.debug;
    success = false;
end

% -------------------------------------------------------------------------
% Extract
% -------------------------------------------------------------------------
phase = struct();
phase.success = success;
phase.phaseName = phaseName;
phase.t = tGrid;
phase.Ncollectors = Nc;
phase.controlChief = setup.controlChief;
phase.constraintStride = constraintStride;

phase.Xwarm = cell(1, setup.Nopt);
phase.Uwarm = cell(1, setup.Nopt);
phase.Xopt = cell(1, setup.Nopt);
phase.Uopt = cell(1, setup.Nopt);
phase.massUsedOpt_kg = zeros(1, setup.Nopt);

for s = 1:setup.Nopt
    phase.Xwarm{s} = S.value(X{s});
    phase.Uwarm{s} = S.value(U{s});
    phase.Xopt{s} = phase.Xwarm{s}.';
    phase.Uopt{s} = phase.Uwarm{s}.';
    phase.massUsedOpt_kg(s) = setup.initStateCell{s}(7) - phase.Xopt{s}(end,7);
end

phase.Xi = S.value(Xi);
phase.objective = S.value(J);
phase.Jfuel = S.value(Jfuel);
phase.Jctrl = S.value(Jctrl);
phase.Jsmooth = S.value(Jsmooth);
phase.Jslack = S.value(Jslack);
phase.massUsedTotal_kg = sum(phase.massUsedOpt_kg);
phase.X = phase.Xopt;
phase.U = phase.Uopt;

end

% =========================================================================
function setup = local_prepare_setup(tGrid, chiefInput, deputyInitCell, params, prob)

Nt = numel(tGrid);
Nc = numel(deputyInitCell);

collectorInitCell = cell(1, Nc);
for j = 1:Nc
    collectorInitCell{j} = local_force_state7(deputyInitCell{j});
end

controlChief = false;
chiefHist = [];
chiefX0 = [];
chiefGuessHist = [];

if isstruct(chiefInput)
    if isfield(chiefInput, 'x0') || (isfield(prob, 'controlChief') && prob.controlChief)
        controlChief = true;
        if isfield(chiefInput, 'x0')
            chiefSeed = chiefInput.x0;
        elseif isfield(chiefInput, 'hist')
            chiefSeed = chiefInput.hist(1,:);
        else
            error('Controlled chief mode requires chiefInput.x0 or chiefInput.hist.');
        end
        chiefX0 = local_force_state7(chiefSeed, local_lookup_chief_mass(prob));

        if isfield(chiefInput, 'guess')
            chiefGuessHist = chiefInput.guess;
        elseif isfield(chiefInput, 'hist')
            chiefGuessHist = chiefInput.hist;
        end
    elseif isfield(chiefInput, 'hist')
        chiefHist = chiefInput.hist;
    else
        error('chiefInput struct must contain either "hist" or "x0".');
    end
elseif isnumeric(chiefInput)
    if isvector(chiefInput)
        controlChief = true;
        chiefX0 = local_force_state7(chiefInput, local_lookup_chief_mass(prob));
    else
        if size(chiefInput, 1) ~= Nt
            error('Numeric chief history must be Nt x 6 or Nt x 7.');
        end

        if isfield(prob, 'controlChief') && prob.controlChief
            controlChief = true;
            chiefX0 = local_force_state7(chiefInput(1,:).', local_lookup_chief_mass(prob));
            chiefGuessHist = chiefInput;
        else
            chiefHist = chiefInput;
        end
    end
else
    error('Unsupported chiefInput format.');
end

if controlChief
    initStateCell = [{chiefX0}, collectorInitCell];
else
    if size(chiefHist, 1) ~= Nt || size(chiefHist, 2) < 6
        error('Legacy fixed-chief mode expects chiefInput as Nt x 6 or Nt x 7.');
    end
    initStateCell = collectorInitCell;
    chiefHist = chiefHist(:,1:6);
end

Nopt = numel(initStateCell);
spacecraftParams = cell(1, Nopt);
for s = 1:Nopt
    spacecraftParams{s} = local_spacecraft_params(params, s, Nopt, Nc, controlChief);
end

massDryKg = local_resolve_dry_mass(prob, Nopt, Nc, controlChief);
[aMinKm, aMaxKm] = local_resolve_a_bounds(prob, params, Nopt, Nc, controlChief);
[coneEnabled, thetaMaxRad] = local_resolve_theta_max(prob, Nopt, Nc, controlChief);

if any(coneEnabled) && ~isfield(prob, 'thrustAxisRTN')
    error('Thrust cone constraints require prob.thrustAxisRTN.');
end

setup = struct();
setup.Nc = Nc;
setup.Nopt = Nopt;
setup.controlChief = controlChief;
setup.initStateCell = initStateCell;
setup.chiefHist = chiefHist;
setup.chiefGuessHist = chiefGuessHist;
setup.spacecraftParams = spacecraftParams;
setup.massDryKg = massDryKg(:).';
setup.aMinKm = aMinKm(:).';
setup.aMaxKm = aMaxKm(:).';
setup.coneEnabled = coneEnabled(:).';
setup.thetaMaxRad = thetaMaxRad(:).';
setup.ephem = params.ephem;

end

% =========================================================================
function F_rk4_cell = local_compile_rk4_functions(h, spacecraftParams)

import casadi.*

Nopt = numel(spacecraftParams);
F_rk4_cell = cell(1, Nopt);

x_cas = MX.sym('x', 7);
u_cas = MX.sym('u', 3);
rSun_cas = MX.sym('rSun', 3);
rMoon_cas = MX.sym('rMoon', 3);

for s = 1:Nopt
    scParams = spacecraftParams{s};

    f1 = scienceHold_QNS_dynamics_casadi(x_cas,            u_cas, scParams, rSun_cas, rMoon_cas);
    f2 = scienceHold_QNS_dynamics_casadi(x_cas + 0.5*h*f1, u_cas, scParams, rSun_cas, rMoon_cas);
    f3 = scienceHold_QNS_dynamics_casadi(x_cas + 0.5*h*f2, u_cas, scParams, rSun_cas, rMoon_cas);
    f4 = scienceHold_QNS_dynamics_casadi(x_cas + h*f3,     u_cas, scParams, rSun_cas, rMoon_cas);
    xnext_expr = x_cas + (h/6) * (f1 + 2*f2 + 2*f3 + f4);

    F_rk4_cell{s} = casadi.Function( ...
        sprintf('F_rk4_%d', s), ...
        {x_cas, u_cas, rSun_cas, rMoon_cas}, ...
        {xnext_expr}, ...
        {'x', 'u', 'rSun', 'rMoon'}, {'xnext'});
end

end

% =========================================================================
function [XguessCell, UguessCell] = local_build_initial_guess(tGrid, rSunTab, rMoonTab, setup)

Nt = numel(tGrid);
Nint = Nt - 1;

XguessCell = cell(1, setup.Nopt);
UguessCell = cell(1, setup.Nopt);

for s = 1:setup.Nopt
    if setup.controlChief && (s == 1) && ~isempty(setup.chiefGuessHist)
        XguessCell{s} = local_prepare_guess_history(setup.chiefGuessHist, setup.initStateCell{s}, Nt);
    else
        XguessCell{s} = local_zero_control_guess( ...
            tGrid, setup.initStateCell{s}, setup.spacecraftParams{s}, rSunTab, rMoonTab);
    end

    UguessCell{s} = zeros(3, Nint);
end

end

% =========================================================================
function Xguess = local_zero_control_guess(tGrid, x0, scParams, rSunTab, rMoonTab)

Nt = numel(tGrid);
Nint = Nt - 1;
h = tGrid(2) - tGrid(1);

Xguess = zeros(7, Nt);
Xguess(:,1) = x0(:);

for k = 1:Nint
    rSun_mid = 0.5 * (rSunTab(k,:).' + rSunTab(k+1,:).');
    rMoon_mid = 0.5 * (rMoonTab(k,:).' + rMoonTab(k+1,:).');
    xk = Xguess(:,k);

    f1 = scienceHold_QNS_dynamics_casadi(xk,            zeros(3,1), scParams, rSun_mid, rMoon_mid);
    f2 = scienceHold_QNS_dynamics_casadi(xk + 0.5*h*f1, zeros(3,1), scParams, rSun_mid, rMoon_mid);
    f3 = scienceHold_QNS_dynamics_casadi(xk + 0.5*h*f2, zeros(3,1), scParams, rSun_mid, rMoon_mid);
    f4 = scienceHold_QNS_dynamics_casadi(xk + h*f3,     zeros(3,1), scParams, rSun_mid, rMoon_mid);

    Xguess(:,k+1) = xk + (h/6) * (f1 + 2*f2 + 2*f3 + f4);
end

end

% =========================================================================
function Xguess = local_prepare_guess_history(guessHist, x0, Nt)

if isempty(guessHist)
    error('local_prepare_guess_history requires a non-empty guess history.');
end

if size(guessHist, 1) == Nt && size(guessHist, 2) >= 6
    guessMat = guessHist;
elseif size(guessHist, 2) == Nt && size(guessHist, 1) >= 6
    guessMat = guessHist.';
else
    error('Chief guess history must be Nt x 6/7 or 6/7 x Nt.');
end

Xguess = zeros(7, Nt);
Xguess(1:6,:) = guessMat(:,1:6).';
if size(guessMat, 2) >= 7
    Xguess(7,:) = guessMat(:,7).';
else
    Xguess(7,:) = x0(7);
end

Xguess(:,1) = x0(:);

end

% =========================================================================
function sol = local_finalize_solution(phase, setup, tGrid, target, prob)

Nt = numel(tGrid);
Nc = setup.Nc;
offset = double(setup.controlChief);

sol = struct();
sol.success = phase.success;
sol.phaseName = phase.phaseName;
sol.t = tGrid;
sol.Ncollectors = Nc;
sol.Nspacecraft = setup.Nopt;
sol.chiefControlled = setup.controlChief;
sol.Dmax_m = prob.Dmax_m;
sol.rmsOPDLimit_m = prob.Dmax_m / 2;

sol.Xall = phase.Xopt;
sol.Uall = phase.Uopt;
sol.massUsedAll_kg = phase.massUsedOpt_kg;

if setup.controlChief
    sol.Xchief = phase.Xopt{1};
    sol.Uchief = phase.Uopt{1};
    sol.chief = sol.Xchief(:,1:6);
    sol.chiefMass_kg = sol.Xchief(:,7);
    sol.massUsedChief_kg = phase.massUsedOpt_kg(1);
else
    sol.Xchief = [];
    sol.Uchief = [];
    sol.chief = setup.chiefHist;
    sol.chiefMass_kg = [];
    sol.massUsedChief_kg = 0;
end

sol.X = cell(1, Nc);
sol.U = cell(1, Nc);
sol.massUsed_kg = zeros(1, Nc);

for j = 1:Nc
    sol.X{j} = phase.Xopt{j + offset};
    sol.U{j} = phase.Uopt{j + offset};
    sol.massUsed_kg(j) = phase.massUsedOpt_kg(j + offset);
end

sol.massUsedCollectorsTotal_kg = sum(sol.massUsed_kg);
sol.massUsedFormation_kg = sum(phase.massUsedOpt_kg);
sol.massUsedTotal_kg = sol.massUsedFormation_kg;

sol.objective = phase.objective;
sol.Jfuel = phase.Jfuel;
sol.Jctrl = phase.Jctrl;
sol.Jsmooth = phase.Jsmooth;
sol.Jslack = phase.Jslack;
sol.xiOPD_km2 = phase.Xi(:).';
sol.xiOPD_m2 = 1e6 * sol.xiOPD_km2;

sol.RTN = cell(1, Nc);
sol.OPD_km = zeros(Nt, Nc);
sol.rangeToChief_km = zeros(Nt, Nc);

for k = 1:Nt
    xChief = sol.chief(k,:).';
    sRTN = full(local_source_rtn_from_target(target, xChief));

    for j = 1:Nc
        xDep = sol.X{j}(k,1:6).';
        dr = full(local_relative_rtn_from_qns(xChief, xDep));

        sol.RTN{j}(k,:) = dr(:).';
        sol.OPD_km(k,j) = dr(:).' * sRTN(:);
        sol.rangeToChief_km(k,j) = norm(dr);
    end
end

sol.OPDspread_km = max(sol.OPD_km, [], 2) - min(sol.OPD_km, [], 2);
sol.OPDspread_m = 1000 * sol.OPDspread_km;
sol.OPDmean_km = mean(sol.OPD_km, 2);
sol.OPDrelative_km = sol.OPD_km - sol.OPDmean_km;
sol.OPDrelative_m = 1000 * sol.OPDrelative_km;

wTrap = local_trapz_weights(tGrid);
Tspan = tGrid(end) - tGrid(1);
weightedRelSq = bsxfun(@times, wTrap, sol.OPDrelative_km.^2);
sol.rmsOPDRelativePerCollector_km = sqrt(sum(weightedRelSq, 1) / Tspan);
sol.rmsOPDRelativePerCollector_m = 1000 * sol.rmsOPDRelativePerCollector_km;
sol.rmsOPDRelativeMax_km = max(sol.rmsOPDRelativePerCollector_km);
sol.rmsOPDRelativeMax_m = 1000 * sol.rmsOPDRelativeMax_km;
sol.rmsOPDConstraintRHS_km = sqrt((prob.Dmax_m / 2000)^2 + sol.xiOPD_km2);
sol.rmsOPDConstraintRHS_m = 1000 * sol.rmsOPDConstraintRHS_km;
sol.rmsOPDConstraintSatisfied = ...
    sol.rmsOPDRelativePerCollector_km <= (sol.rmsOPDConstraintRHS_km + 1e-9);

sol.thrustNorm = cell(1, Nc);
for j = 1:Nc
    sol.thrustNorm{j} = sqrt(sum(sol.U{j}.^2, 2));
end

if setup.controlChief
    sol.thrustNormChief = sqrt(sum(sol.Uchief.^2, 2));
else
    sol.thrustNormChief = [];
end

sol.thrustNormAll = cell(1, setup.Nopt);
for s = 1:setup.Nopt
    sol.thrustNormAll{s} = sqrt(sum(sol.Uall{s}.^2, 2));
end

nPairs = Nc * (Nc - 1) / 2;
sol.pairSep_km = zeros(Nt, nPairs);
sol.pairSepLabels = cell(1, nPairs);
pairIdx = 0;
for i = 1:Nc-1
    for j = i+1:Nc
        pairIdx = pairIdx + 1;
        sol.pairSepLabels{pairIdx} = sprintf('%d-%d', i, j);
        sol.pairSep_km(:,pairIdx) = sqrt(sum((sol.RTN{i} - sol.RTN{j}).^2, 2));
    end
end

dvAll = zeros(1, setup.Nopt);
for s = 1:setup.Nopt
    Tsc = local_param_with_default(setup.spacecraftParams{s}, 'T', 0);
    mNodes = sol.Xall{s}(:,7);
    uNorm = sol.thrustNormAll{s};
    mMid = 0.5 * (mNodes(1:end-1) + mNodes(2:end));
    aMid = (Tsc ./ mMid) .* uNorm;
    dvAll(s) = sum(aMid .* diff(tGrid));
end

sol.dvApproxAll_mps = dvAll;
if setup.controlChief
    sol.dvApproxChief_mps = dvAll(1);
    sol.dvApprox_mps = dvAll(2:end);
else
    sol.dvApproxChief_mps = 0;
    sol.dvApprox_mps = dvAll;
end
sol.dvApproxTotal_mps = sum(dvAll);

end

% =========================================================================
function scParams = local_spacecraft_params(params, scIdx, Nopt, Nc, controlChief)

scParams = params;
containerFields = {'ephem', 'sc', 'chief', 'collectors'};
for k = 1:numel(containerFields)
    if isfield(scParams, containerFields{k})
        scParams = rmfield(scParams, containerFields{k});
    end
end

perSpacecraftFields = {'T', 'Isp', 'As', 'CR'};
for k = 1:numel(perSpacecraftFields)
    f = perSpacecraftFields{k};
    if isfield(scParams, f)
        scParams.(f) = local_pick_spacecraft_value(scParams.(f), scIdx, Nopt, Nc, controlChief, f);
    end
end

override = struct();
if isfield(params, 'sc')
    override = local_pick_container_entry(params.sc, scIdx);
elseif controlChief && (scIdx == 1) && isfield(params, 'chief')
    override = local_pick_container_entry(params.chief, 1);
elseif controlChief && (scIdx > 1) && isfield(params, 'collectors')
    override = local_pick_container_entry(params.collectors, scIdx - 1);
end

scParams = local_merge_structs(scParams, override);

for k = 1:numel(perSpacecraftFields)
    f = perSpacecraftFields{k};
    if isfield(scParams, f)
        scParams.(f) = local_pick_spacecraft_value(scParams.(f), scIdx, Nopt, Nc, controlChief, f);
    end
end

end

% =========================================================================
function value = local_pick_spacecraft_value(value, scIdx, Nopt, Nc, controlChief, fieldName)

if ~isnumeric(value) || isscalar(value)
    return;
end

if isvector(value)
    value = value(:);
    if numel(value) == Nopt
        value = value(scIdx);
    elseif (~controlChief) && (numel(value) == Nc)
        value = value(scIdx);
    elseif controlChief && (numel(value) == Nc) && (scIdx > 1)
        value = value(scIdx - 1);
    else
        error('Cannot resolve per-spacecraft field "%s" for spacecraft %d.', fieldName, scIdx);
    end
end

end

% =========================================================================
function entry = local_pick_container_entry(container, idx)

if isempty(container)
    entry = struct();
    return;
end

if iscell(container)
    if isscalar(container)
        entry = container{1};
    elseif numel(container) >= idx
        entry = container{idx};
    else
        error('Container entry %d is missing.', idx);
    end
elseif isstruct(container)
    if isscalar(container)
        entry = container;
    elseif numel(container) >= idx
        entry = container(idx);
    else
        error('Struct entry %d is missing.', idx);
    end
else
    error('Unsupported spacecraft-parameter container type.');
end

if isempty(entry)
    entry = struct();
end

end

% =========================================================================
function merged = local_merge_structs(base, override)

merged = base;
if ~isstruct(override)
    return;
end

overrideFields = fieldnames(override);
for k = 1:numel(overrideFields)
    merged.(overrideFields{k}) = override.(overrideFields{k});
end

end

% =========================================================================
function massDryKg = local_resolve_dry_mass(prob, Nopt, Nc, controlChief)

massDryKg = local_expand_per_spacecraft(prob.massDry_kg, Nopt, Nc, controlChief, 'prob.massDry_kg');

if controlChief
    if isfield(prob, 'massDryChief_kg') && ~isempty(prob.massDryChief_kg)
        massDryKg(1) = prob.massDryChief_kg;
    end

    if any(~isfinite(massDryKg))
        error('Provide prob.massDryChief_kg when prob.massDry_kg only covers the collectors.');
    end
end

end

% =========================================================================
function [aMinKm, aMaxKm] = local_resolve_a_bounds(prob, params, Nopt, Nc, controlChief)

aFloorDefault = max(local_param_with_default(params, 'RE', 0) + 1, 1);

if isfield(prob, 'aMin_km') && ~isempty(prob.aMin_km)
    aMinKm = local_expand_per_spacecraft(prob.aMin_km, Nopt, Nc, controlChief, 'prob.aMin_km');
else
    aMinKm = repmat(aFloorDefault, 1, Nopt);
end

if isfield(prob, 'aMax_km') && ~isempty(prob.aMax_km)
    aMaxKm = local_expand_per_spacecraft(prob.aMax_km, Nopt, Nc, controlChief, 'prob.aMax_km');
else
    aMaxKm = inf(1, Nopt);
end

end

% =========================================================================
function [coneEnabled, thetaMaxRad] = local_resolve_theta_max(prob, Nopt, Nc, controlChief)

coneEnabled = false(1, Nopt);
thetaMaxRad = zeros(1, Nopt);

baseTheta = [];
if isfield(prob, 'thetaMax_rad')
    baseTheta = prob.thetaMax_rad;
elseif isfield(prob, 'thrustConeHalfAngle_rad')
    baseTheta = prob.thrustConeHalfAngle_rad;
elseif isfield(prob, 'thetaMax_deg')
    baseTheta = deg2rad(prob.thetaMax_deg);
elseif isfield(prob, 'thrustConeHalfAngle_deg')
    baseTheta = deg2rad(prob.thrustConeHalfAngle_deg);
end

if ~isempty(baseTheta)
    thetaBase = local_expand_per_spacecraft(baseTheta, Nopt, Nc, controlChief, 'theta max');
    finiteMask = isfinite(thetaBase);
    thetaMaxRad(finiteMask) = thetaBase(finiteMask);
    coneEnabled(finiteMask) = true;
end

if controlChief
    if isfield(prob, 'thetaMaxChief_rad') && ~isempty(prob.thetaMaxChief_rad)
        thetaMaxRad(1) = prob.thetaMaxChief_rad;
        coneEnabled(1) = true;
    elseif isfield(prob, 'thetaMaxChief_deg') && ~isempty(prob.thetaMaxChief_deg)
        thetaMaxRad(1) = deg2rad(prob.thetaMaxChief_deg);
        coneEnabled(1) = true;
    end

    if isfield(prob, 'thetaMaxCollector_rad') && ~isempty(prob.thetaMaxCollector_rad)
        collectorTheta = local_expand_collector_value(prob.thetaMaxCollector_rad, Nc, 'thetaMaxCollector_rad');
        thetaMaxRad(2:end) = collectorTheta;
        coneEnabled(2:end) = true;
    elseif isfield(prob, 'thetaMaxCollector_deg') && ~isempty(prob.thetaMaxCollector_deg)
        collectorTheta = local_expand_collector_value(deg2rad(prob.thetaMaxCollector_deg), Nc, 'thetaMaxCollector_deg');
        thetaMaxRad(2:end) = collectorTheta;
        coneEnabled(2:end) = true;
    end
end

end

% =========================================================================
function values = local_expand_per_spacecraft(value, Nopt, Nc, controlChief, label)

if isscalar(value)
    values = repmat(double(value), 1, Nopt);
    return;
end

value = value(:).';
if numel(value) == Nopt
    values = double(value);
    return;
end

if (~controlChief) && (numel(value) == Nc)
    values = double(value);
    return;
end

if controlChief && (numel(value) == Nc)
    values = [nan, double(value)];
    return;
end

error('Cannot expand %s to %d spacecraft.', label, Nopt);

end

% =========================================================================
function values = local_expand_collector_value(value, Nc, label)

if isscalar(value)
    values = repmat(double(value), 1, Nc);
elseif numel(value) == Nc
    values = double(value(:)).';
else
    error('%s must be scalar or length Nc.', label);
end

end

% =========================================================================
function mass0 = local_lookup_chief_mass(prob)

if isfield(prob, 'chiefMass0_kg') && ~isempty(prob.chiefMass0_kg)
    mass0 = prob.chiefMass0_kg;
elseif isfield(prob, 'mass0Chief_kg') && ~isempty(prob.mass0Chief_kg)
    mass0 = prob.mass0Chief_kg;
else
    mass0 = [];
end

end

% =========================================================================
function x = local_force_state7(xIn, fallbackMass)

if nargin < 2
    fallbackMass = [];
end

xIn = xIn(:);

if numel(xIn) == 7
    x = xIn;
elseif numel(xIn) == 6
    if isempty(fallbackMass)
        error('A 6-state spacecraft input requires an explicit initial mass.');
    end
    x = [xIn; fallbackMass];
else
    error('Spacecraft states must contain 6 or 7 elements.');
end

end

% =========================================================================
function idx = local_sample_indices(Nt, stride)

stride = max(1, round(stride));
idx = 1:stride:Nt;
if idx(end) ~= Nt
    idx = [idx, Nt];
end
idx = unique(idx, 'stable');

end

% =========================================================================
function w = local_trapz_weights(t)

t = t(:);
N = numel(t);
w = zeros(N, 1);

if N == 1
    return;
end

w(1) = 0.5 * (t(2) - t(1));
for k = 2:N-1
    w(k) = 0.5 * (t(k+1) - t(k-1));
end
w(N) = 0.5 * (t(N) - t(N-1));

end

% =========================================================================
function uMag = local_smooth_norm(u)
uMag = sqrt(sumsqr(u) + 1e-10^2) - 1e-10;
end

% =========================================================================
function bHat = local_get_thrust_axis_rtn(prob, scIdx, k, Nopt, Nc, controlChief, Nint, Nt)

axisData = prob.thrustAxisRTN;

if iscell(axisData)
    if isscalar(axisData)
        axisSpec = axisData{1};
    elseif numel(axisData) == Nopt
        axisSpec = axisData{scIdx};
    elseif controlChief && (numel(axisData) == Nc)
        if scIdx == 1
            error('Chief thrust axis is missing from prob.thrustAxisRTN.');
        end
        axisSpec = axisData{scIdx - 1};
    else
        error('Unsupported cell-array length for prob.thrustAxisRTN.');
    end
else
    axisSpec = axisData;
end

bHat = local_resolve_axis_spec(axisSpec, scIdx, k, Nopt, Nc, controlChief, Nint, Nt);
bHat = local_normalize_axis(bHat);

end

% =========================================================================
function axisVec = local_resolve_axis_spec(axisSpec, scIdx, k, Nopt, Nc, controlChief, Nint, Nt)

if isvector(axisSpec) && numel(axisSpec) == 3
    axisVec = axisSpec(:);
    return;
end

if ~isnumeric(axisSpec) || ~ismatrix(axisSpec)
    error('Each thrust-axis specification must be numeric and 2-D.');
end

[nr, nc] = size(axisSpec);

if (nr == 3) && (nc == Nopt)
    axisVec = axisSpec(:, scIdx);
elseif controlChief && (nr == 3) && (nc == Nc)
    if scIdx == 1
        error('Chief thrust axis is missing from prob.thrustAxisRTN.');
    end
    axisVec = axisSpec(:, scIdx - 1);
elseif (nr == Nopt) && (nc == 3)
    axisVec = axisSpec(scIdx,:).';
elseif controlChief && (nr == Nc) && (nc == 3)
    if scIdx == 1
        error('Chief thrust axis is missing from prob.thrustAxisRTN.');
    end
    axisVec = axisSpec(scIdx - 1,:).';
elseif (nr == 3) && (nc == Nint)
    axisVec = axisSpec(:, min(k, Nint));
elseif (nr == Nint) && (nc == 3)
    axisVec = axisSpec(min(k, Nint), :).';
elseif (nr == 3) && (nc == Nt)
    axisVec = axisSpec(:, min(k, Nt));
elseif (nr == Nt) && (nc == 3)
    axisVec = axisSpec(min(k, Nt), :).';
else
    error('Unsupported prob.thrustAxisRTN shape.');
end

end

% =========================================================================
function bHat = local_normalize_axis(bAxis)

bAxis = double(bAxis(:));
nrm = norm(bAxis);
if ~(nrm > 0)
    error('Each thrust axis must have non-zero magnitude.');
end
bHat = bAxis / nrm;

end

% =========================================================================
function sRTN = local_source_rtn_from_target(target, chiefState)

inc = chiefState(4);
RAAN = chiefState(5);
u = chiefState(6);

if strcmpi(target.type, 'phibeta')
    phi0 = target.phi0;
    beta = target.beta;
    sRTN = [cos(beta) * cos(phi0 - u);
            cos(beta) * sin(phi0 - u);
            sin(beta)];
elseif strcmpi(target.type, 'radec')
    sRTN = local_source_rtn_from_radec(target.ra, target.dec, inc, RAAN, u);
else
    error('Unknown target.type.');
end

end

% =========================================================================
function dr = local_relative_rtn_from_qns(xChief, xDeputy)

ac = xChief(1);
exc = xChief(2);
eyc = xChief(3);
ic = xChief(4);
RAc = xChief(5);
uc = xChief(6);

ad = xDeputy(1);
exd = xDeputy(2);
eyd = xDeputy(3);
id = xDeputy(4);
RAd = xDeputy(5);
ud = xDeputy(6);

dRA = RAd - RAc;
du = ud - uc;

delta_a = (ad - ac) / ac;
delta_lam = du + dRA * cos(ic);
delta_ex = exd - exc;
delta_ey = eyd - eyc;
delta_ix = id - ic;
delta_iy = dRA * sin(ic);

dR = ac * (delta_a - cos(uc) * delta_ex - sin(uc) * delta_ey);
dT = ac * (delta_lam + 2 * sin(uc) * delta_ex - 2 * cos(uc) * delta_ey);
dN = ac * (sin(uc) * delta_ix - cos(uc) * delta_iy);

dr = [dR; dT; dN];

end

% =========================================================================
function value = local_param_with_default(params, fieldName, defaultValue)

if isfield(params, fieldName) && ~isempty(params.(fieldName))
    value = params.(fieldName);
else
    value = defaultValue;
end

end

% =========================================================================
function sRTN = local_source_rtn_from_radec(ra, dec, inc, RAAN, u)
sI = [cos(dec) * cos(ra);
      cos(dec) * sin(ra);
      sin(dec)];

cO = cos(RAAN); sO = sin(RAAN);
ci = cos(inc);  si = sin(inc);
cu = cos(u);    su = sin(u);

Rhat = [ cO*cu - sO*su*ci;
         sO*cu + cO*su*ci;
         su*si ];

That = [ -cO*su - sO*cu*ci;
         -sO*su + cO*cu*ci;
          cu*si ];

Nhat = [-sO*si;
         cO*si;
         -ci ];

sRTN = [dot(sI, Rhat);
        dot(sI, That);
        dot(sI, Nhat)];
end
