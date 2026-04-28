function sol = solve_science_hold_ocp(tGrid, chiefHist, deputyInitCell, params, target, prob)
% solve_science_hold_ocp
%
% Fixed-duration science-hold OCP for N collector satellites around a
% precomputed chief trajectory.
%
% Key features:
%   - dynamically consistent initial guess
%   - soft OPD constraints with slack
%   - two-phase solve:
%       Phase 1: minimize slack + smoothness
%       Phase 2: minimize fuel + light regularization
%
% Inputs:
%   tGrid          : (Nt x 1) time grid [s]
%   chiefHist      : (Nt x 6) chief QNS reference trajectory
%   deputyInitCell : 1xNc cell array, each 7x1 initial state:
%                    [a; ex; ey; inc; RAAN; u; m]
%   params         : struct with dynamics/propulsion params
%   target         : struct with target definition
%   prob           : struct with OCP settings
%
% Required prob fields:
%   prob.Dmax_m
%   prob.rhoMin_km
%   prob.dPairMin_km
%   prob.massDry_kg
%
% Optional prob fields:
%   prob.rhoMax_km
%   prob.phase1ConstraintStride   default 5
%   prob.phase1_wSlack            default 1
%   prob.phase1_wSmooth           default 1e-4
%   prob.phase2_wSlack            default 1e3
%   prob.phase2_wFuel             default 1
%   prob.phase2_wControl          default 1e-5
%   prob.phase2_wSmooth           default 1e-4
%

import casadi.*

% ---------------- Defaults ----------------
if ~isfield(prob,'phase1ConstraintStride'), prob.phase1ConstraintStride = 5; end
if ~isfield(prob,'phase1_wSlack'),          prob.phase1_wSlack = 1; end
if ~isfield(prob,'phase1_wSmooth'),         prob.phase1_wSmooth = 1e-4; end

if ~isfield(prob,'phase2_wSlack'),          prob.phase2_wSlack = 1e1; end
if ~isfield(prob,'phase2_wFuel'),           prob.phase2_wFuel = 1; end
if ~isfield(prob,'phase2_wControl'),        prob.phase2_wControl = 1e-5; end
if ~isfield(prob,'phase2_wSmooth'),         prob.phase2_wSmooth = 1e-4; end

Nc = numel(deputyInitCell);
Nt = numel(tGrid);
Nint = Nt - 1;

dt = diff(tGrid(:));
if max(abs(dt - dt(1))) > 1e-9
    error('tGrid must be uniform for this version.');
end
h = dt(1);

xc = chiefHist;  % chief reference trajectory

rSunTab  = params.ephem.rSun;   % Nt x 3
rMoonTab = params.ephem.rMoon;  % Nt x 3

% =========================================================================
% Compile CasADi RK4 Function 
% =========================================================================
fprintf('  Compiling RK4 CasADi Function...');
x_cas    = casadi.MX.sym('x',    7);
u_cas    = casadi.MX.sym('u',    3);
rSun_cas = casadi.MX.sym('rSun', 3);
rMoon_cas= casadi.MX.sym('rMoon',3);
 
f1 = scienceHold_QNS_dynamics_casadi(x_cas,              u_cas, params, rSun_cas, rMoon_cas);
f2 = scienceHold_QNS_dynamics_casadi(x_cas + 0.5*h*f1,   u_cas, params, rSun_cas, rMoon_cas);
f3 = scienceHold_QNS_dynamics_casadi(x_cas + 0.5*h*f2,   u_cas, params, rSun_cas, rMoon_cas);
f4 = scienceHold_QNS_dynamics_casadi(x_cas + h*f3,       u_cas, params, rSun_cas, rMoon_cas);
xnext_expr = x_cas + (h/6)*(f1 + 2*f2 + 2*f3 + f4);
 
F_rk4 = casadi.Function('F_rk4', ...
    {x_cas, u_cas, rSun_cas, rMoon_cas}, ...
    {xnext_expr}, ...
    {'x','u','rSun','rMoon'}, {'xnext'});
fprintf(' done.\n');

% =========================================================================
% Build a dynamically consistent initial guess (zero control propagation)
% =========================================================================
XguessCell = cell(1,Nc);
UguessCell = cell(1,Nc);

for j = 1:Nc
    x0j = deputyInitCell{j}(:);
    Xguess = zeros(7, Nt);
    Xguess(:,1) = x0j;

    for k = 1:Nint
        rSun_mid  = 0.5*(rSunTab(k,:).'  + rSunTab(k+1,:).');
        rMoon_mid = 0.5*(rMoonTab(k,:).' + rMoonTab(k+1,:).');
        xk = Xguess(:,k);

        % reuse same dynamics with numeric doubles
        f1 = scienceHold_QNS_dynamics_casadi(xk, zeros(3,1), params, rSun_mid, rMoon_mid);
        f2 = scienceHold_QNS_dynamics_casadi(xk + 0.5*h*f1, zeros(3,1), params, rSun_mid, rMoon_mid);
        f3 = scienceHold_QNS_dynamics_casadi(xk + 0.5*h*f2, zeros(3,1), params, rSun_mid, rMoon_mid);
        f4 = scienceHold_QNS_dynamics_casadi(xk + h*f3,     zeros(3,1), params, rSun_mid, rMoon_mid);

        Xguess(:,k+1) = xk + (h/6)*(f1 + 2*f2 + 2*f3 + f4);
    end

    XguessCell{j} = Xguess;
    UguessCell{j} = zeros(3,Nint);
end

% =========================================================================
% Phase 1 solve
% =========================================================================
phase1 = build_and_solve_phase( ...
    'phase1', tGrid, xc, deputyInitCell, params, target, prob, F_rk4,...
    XguessCell, UguessCell, ...
    prob.phase1ConstraintStride, ...
    prob.phase1_wSlack, 0, 0, prob.phase1_wSmooth);

for ii=1:numel(phase1.X)
phase1.X{ii} = phase1.X{ii}';
phase1.U{ii} = phase1.U{ii}';
end
% =========================================================================
% Phase 2 solve (full constraints, warm start)
% =========================================================================
phase2 = build_and_solve_phase( ...
    'phase2', tGrid, xc, deputyInitCell, params, target, prob, F_rk4, ...
    phase1.X, phase1.U, ...
    1, ...
    prob.phase2_wSlack, prob.phase2_wFuel, prob.phase2_wControl, prob.phase2_wSmooth);

% =========================================================================
% Final solution packaging
% =========================================================================
sol = phase2;
sol.phase1 = phase1;
sol.phase2 = phase2;

end

% =========================================================================
function sol = build_and_solve_phase(phaseName, tGrid, chiefHist, deputyInitCell, params, target, prob, F_rk4,...
                                     XinitCell, UinitCell, constraintStride, wSlack, wFuel, wControl, wSmooth)

import casadi.*

Nc = numel(deputyInitCell);
Nt = numel(tGrid);
Nint = Nt - 1;
h = tGrid(2) - tGrid(1);

xc = chiefHist;
rSunTab  = params.ephem.rSun;
rMoonTab = params.ephem.rMoon;

Dtol_km = (prob.Dmax_m/1000)/2;

opti = casadi.Opti();

X = cell(1,Nc);
U = cell(1,Nc);
for j = 1:Nc
    X{j} = opti.variable(7, Nt);
    U{j} = opti.variable(3, Nint);
end

% Slack on OPD constraints
Sopd = opti.variable(Nc, Nt);
opti.subject_to(vec(Sopd) >= 0);


% -------------------------------------------------------------------------
% Initial conditions
% -------------------------------------------------------------------------
for j = 1:Nc
    opti.subject_to(X{j}(:,1) == deputyInitCell{j}(:));
end

% -------------------------------------------------------------------------
% Dynamics constraints  <-- uses compiled F_rk4, O(N) graph construction
% -------------------------------------------------------------------------
for k = 1:Nint
    rSm = 0.5*(rSunTab(k,:).'  + rSunTab(k+1,:).');
    rMm = 0.5*(rMoonTab(k,:).' + rMoonTab(k+1,:).');
 
    for j = 1:Nc
        xnext = F_rk4(X{j}(:,k), U{j}(:,k), rSm, rMm);
        opti.subject_to(X{j}(:,k+1) == xnext);
    end
end


% -------------------------------------------------------------------------
% Control and mass bounds
% -------------------------------------------------------------------------
for j = 1:Nc
    for k = 1:Nint
        opti.subject_to(sumsqr(U{j}(:,k)) <= 1.0);
    end
    opti.subject_to(vec(X{j}(7,:).') >= prob.massDry_kg);
    % This directly enforces physical propellant conservation
    opti.subject_to(X{j}(7,2:end) <= X{j}(7,1:end-1));

end

% -------------------------------------------------------------------------
% Path constraints
% -------------------------------------------------------------------------
for k = 1:constraintStride:Nt
    opd_k = MX.zeros(Nc,1);
    drMat = MX.zeros(3,Nc);

    ac    = xc(k,1);
    exc   = xc(k,2);
    eyc   = xc(k,3);
    ic    = xc(k,4);
    RAc   = xc(k,5);
    uc    = xc(k,6);

    if strcmpi(target.type,'phibeta')
        phi0 = target.phi0;
        beta = target.beta;
        sRTN = [cos(beta)*cos(phi0 - uc);
                cos(beta)*sin(phi0 - uc);
                sin(beta)];
    elseif strcmpi(target.type,'radec')
        sRTN = local_source_rtn_from_radec(target.ra, target.dec, ic, RAc, uc);
    else
        error('Unknown target.type');
    end

    for j = 1:Nc
        ad  = X{j}(1,k);
        exd = X{j}(2,k);
        eyd = X{j}(3,k);
        id  = X{j}(4,k);
        RAd = X{j}(5,k);
        ud  = X{j}(6,k);

        dRA = RAd - RAc;
        du  = ud  - uc;

        delta_a   = (ad - ac)/ac;
        delta_lam = du + dRA*cos(ic);
        delta_ex  = exd - exc;
        delta_ey  = eyd - eyc;
        delta_ix  = id  - ic;
        delta_iy  = dRA*sin(ic);

        dR = ac*( delta_a - cos(uc)*delta_ex - sin(uc)*delta_ey );
        dT = ac*( delta_lam + 2*sin(uc)*delta_ex - 2*cos(uc)*delta_ey );
        dN = ac*( sin(uc)*delta_ix - cos(uc)*delta_iy );

        dr = [dR; dT; dN];
        drMat(:,j) = dr;
        opd_k(j) = dr.' * sRTN;
    end

    opdMean = sum1(opd_k)/Nc;

    % Soft OPD spread constraint
    for j = 1:Nc
        opti.subject_to(opd_k(j) - opdMean <=  Dtol_km + Sopd(j,k));
        opti.subject_to(opd_k(j) - opdMean >= -Dtol_km - Sopd(j,k));
        
    end

    % Baseline preservation
    for j = 1:Nc
        rj2 = sumsqr(drMat(:,j));
        opti.subject_to(rj2 >= prob.rhoMin_km^2);
        if isfield(prob,'rhoMax_km') && ~isempty(prob.rhoMax_km)
            opti.subject_to(rj2 <= prob.rhoMax_km^2);
        end
    end

    % Pairwise separation
    for i = 1:Nc-1
        for j = i+1:Nc
            dij2 = sumsqr(drMat(:,i) - drMat(:,j));
            opti.subject_to(dij2 >= prob.dPairMin_km^2);
        end
    end
end

% -------------------------------------------------------------------------
% Objective
% -------------------------------------------------------------------------
Jfuel = 0;
Jctrl = 0;
Jsmooth = 0;

for j = 1:Nc
    m0 = deputyInitCell{j}(7);
    mf = X{j}(7,end);

    Jfuel = Jfuel + (m0 - mf);

    for k = 1:Nint
        Jctrl = Jctrl + wControl * h * sumsqr(U{j}(:,k));
    end

    for k = 1:Nint-1
        Jsmooth = Jsmooth + wSmooth * sumsqr(U{j}(:,k+1) - U{j}(:,k));
    end
end

Jslack = wSlack * sum1(sum2(Sopd));

J = wFuel*Jfuel + Jctrl + Jsmooth + Jslack;
opti.minimize(J);

% -------------------------------------------------------------------------
% Initial guess
% -------------------------------------------------------------------------
for j = 1:Nc
    opti.set_initial(X{j}, XinitCell{j});
    opti.set_initial(U{j}, UinitCell{j});
end
opti.set_initial(Sopd, 0);

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
sol = struct();
sol.success = success;
sol.phaseName = phaseName;
sol.t = tGrid;
sol.chief = chiefHist;
sol.Ncollectors = Nc;

sol.X = cell(1,Nc);
sol.U = cell(1,Nc);
sol.massUsed_kg = zeros(1,Nc);

for j = 1:Nc
    sol.X{j} = S.value(X{j}).';
    sol.U{j} = S.value(U{j}).';
    sol.massUsed_kg(j) = deputyInitCell{j}(7) - sol.X{j}(end,7);
end

sol.Sopd = S.value(Sopd);
sol.maxSlack_km = max(sol.Sopd(:));
sol.maxSlack_m  = 1000*sol.maxSlack_km;

sol.objective = S.value(J);
sol.Jfuel = S.value(Jfuel);
sol.Jctrl = S.value(Jctrl);
sol.Jsmooth = S.value(Jsmooth);
sol.Jslack = S.value(Jslack);

sol.massUsedTotal_kg = sum(sol.massUsed_kg);

% Recompute OPD spread from solution
sol.OPD_km = zeros(Nt,Nc);
sol.OPDspread_km = zeros(Nt,1);



for k = 1:Nt
    ac    = chiefHist(k,1);
    exc   = chiefHist(k,2);
    eyc   = chiefHist(k,3);
    ic    = chiefHist(k,4);
    RAc   = chiefHist(k,5);
    uc    = chiefHist(k,6);

    if strcmpi(target.type,'phibeta')
        phi0 = target.phi0;
        beta = target.beta;
        sRTN = [cos(beta)*cos(phi0 - uc);
                cos(beta)*sin(phi0 - uc);
                sin(beta)];
    else
        sRTN = full(local_source_rtn_from_radec(target.ra, target.dec, ic, RAc, uc));
    end

    for j = 1:Nc
        xd = sol.X{j}(k,:).';

        ad  = xd(1);
        exd = xd(2);
        eyd = xd(3);
        id  = xd(4);
        RAd = xd(5);
        ud  = xd(6);

        dRA = RAd - RAc;
        du  = ud  - uc;

        delta_a   = (ad - ac)/ac;
        delta_lam = du + dRA*cos(ic);
        delta_ex  = exd - exc;
        delta_ey  = eyd - eyc;
        delta_ix  = id  - ic;
        delta_iy  = dRA*sin(ic);

        dR = ac*( delta_a - cos(uc)*delta_ex - sin(uc)*delta_ey );
        dT = ac*( delta_lam + 2*sin(uc)*delta_ex - 2*cos(uc)*delta_ey );
        dN = ac*( sin(uc)*delta_ix - cos(uc)*delta_iy );
        sol.RTN{j}(k,:) = [dR dT dN];
        sol.OPD_km(k,j) = [dR dT dN] * full(sRTN(:));
    end

    sol.OPDspread_km(k) = max(sol.OPD_km(k,:)) - min(sol.OPD_km(k,:));
end

for jj=1:numel(Nc)
   sol.rangeToChief_km(:,jj) = sqrt(sum(sol.RTN{jj}.^2,2));
end
sol.OPDspread_m = 1000*sol.OPDspread_km;
sol.OPDmean_km = mean(sol.OPD_km, 2);
sol.OPDrelative_km = sol.OPD_km - sol.OPDmean_km;
% -------------------------------------------------------------------------
% E) Pairwise separations
% -------------------------------------------------------------------------
nPairs = Nc*(Nc-1)/2;
sol.pairSep_km = zeros(Nt, nPairs);
sol.pairSepLabels = cell(1, nPairs);

pairIdx = 0;

% If sol.RTN exists as cell or 3D array, use it; otherwise compute from X/chief
if isfield(sol,'RTN')
    useExistingRTN = true;
else
    useExistingRTN = false;
end

for i = 1:Nc-1
    for j = i+1:Nc
        pairIdx = pairIdx + 1;
        sol.pairSepLabels{pairIdx} = sprintf('%d-%d', i, j);

        for k = 1:Nt
            if useExistingRTN
                if iscell(sol.RTN)
                    ri = sol.RTN{i}(k,:).';
                    rj = sol.RTN{j}(k,:).';
                else
                    ri = squeeze(sol.RTN(k,:,i)).';
                    rj = squeeze(sol.RTN(k,:,j)).';
                end
            else
                % Compute RTN from states
                ac    = chiefHist(k,1);
                exc   = chiefHist(k,2);
                eyc   = chiefHist(k,3);
                ic    = chiefHist(k,4);
                RAc   = chiefHist(k,5);
                uc    = chiefHist(k,6);

                xi = sol.X{i}(k,:).';
                xj = sol.X{j}(k,:).';

                % deputy i
                dRAi = xi(5) - RAc;
                dui  = xi(6) - uc;
                dai  = (xi(1) - ac)/ac;
                dli  = dui + dRAi*cos(ic);
                dexi = xi(2) - exc;
                deyi = xi(3) - eyc;
                dixi = xi(4) - ic;
                diyi = dRAi*sin(ic);

                ri = [ ac*( dai - cos(uc)*dexi - sin(uc)*deyi );
                       ac*( dli + 2*sin(uc)*dexi - 2*cos(uc)*deyi );
                       ac*( sin(uc)*dixi - cos(uc)*diyi ) ];

                % deputy j
                dRAj = xj(5) - RAc;
                duj  = xj(6) - uc;
                daj  = (xj(1) - ac)/ac;
                dlj  = duj + dRAj*cos(ic);
                dexj = xj(2) - exc;
                deyj = xj(3) - eyc;
                dixj = xj(4) - ic;
                diyj = dRAj*sin(ic);

                rj = [ ac*( daj - cos(uc)*dexj - sin(uc)*deyj );
                       ac*( dlj + 2*sin(uc)*dexj - 2*cos(uc)*deyj );
                       ac*( sin(uc)*dixj - cos(uc)*diyj ) ];
            end

            sol.pairSep_km(k, pairIdx) = norm(ri - rj);
        end
    end
end

% -------------------------------------------------------------------------
% F) Slack per collector
% -------------------------------------------------------------------------
% Sopd is stored as Nc x Nt
if isfield(sol,'Sopd')
    sol.slackPerCollector_km = sol.Sopd.';          % Nt x Nc
    sol.slackMax_km = max(sol.slackPerCollector_km, [], 2);  % Nt x 1
    sol.slackPerCollector_m = 1000 * sol.slackPerCollector_km;
    sol.slackMax_m = 1000 * sol.slackMax_km;
end

% -------------------------------------------------------------------------
% G) Thrust norm
% -------------------------------------------------------------------------
sol.thrustNorm = cell(1,Nc);
for j = 1:Nc
    Uj = sol.U{j};   % (Nt-1) x 3
    sol.thrustNorm{j} = sqrt(sum(Uj.^2, 2));   % (Nt-1) x 1
end

% -------------------------------------------------------------------------
% H) Approximate delta-v from thrust history
% -------------------------------------------------------------------------
% dvApprox = integral (T/m * ||u|| dt)
% Uses midpoint/interval mass approximation from node values
sol.dvApprox_mps = zeros(1,Nc);

for j = 1:Nc
    mj_nodes = sol.X{j}(:,7);          % Nt x 1
    uNorm_j  = sol.thrustNorm{j};      % Nt-1 x 1

    % midpoint mass over each interval
    m_mid = 0.5 * (mj_nodes(1:end-1) + mj_nodes(2:end));

    % acceleration magnitude in m/s^2
    a_mid = (params.T ./ m_mid) .* uNorm_j;

    % integrate over time
    sol.dvApprox_mps(j) = sum(a_mid .* diff(tGrid(:)));
end

sol.dvApproxTotal_mps = sum(sol.dvApprox_mps);


end

% =========================================================================
function sRTN = local_source_rtn_from_radec(ra, dec, inc, RAAN, u)
sI = [cos(dec)*cos(ra);
      cos(dec)*sin(ra);
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

sRTN = [dot(sI,Rhat);
        dot(sI,That);
        dot(sI,Nhat)];
end