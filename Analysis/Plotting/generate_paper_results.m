function R = generate_paper_results(out, sol, varargin)
%GENERATE_PAPER_RESULTS  Journal-format tables and figures for the GEO
%   formation-flying interferometer paper (SPIE JATIS style).
%
%   R = generate_paper_results(out, sol)
%   R = generate_paper_results({out1,...,outN}, {sol1,...,solN}, 'Name', Value, ...)
%
%   INPUTS
%     out : struct (or cell array of structs) from Example_GEO_Formation_OPD
%     sol : struct (or cell array of structs) from the OCP solve. If several
%           sols share one out (same target, different T_int), pass a single out.
%
%   NAME-VALUE OPTIONS (defaults in brackets)
%     'OutDir'          ['paper_results']  output folder
%     'Labels'          {}  one label per case, e.g. {'Altair, 1 h', ...}
%     'TargetNames'     {}  one name per case (otherwise looked up from RA/Dec)
%     'Dmax_m'          [5]       OPD limit used in the OCP [m]
%     'ConstraintTol_m' [0.1]     tolerance for "bound met" (IPOPT constr_viol_tol 1e-4 km)
%     'T_N'             [0.04]    maximum thrust [N]
%     'Isp_s'           [1500]    specific impulse [s]
%     'RE_km'           [6378]    Earth radius for the shadow test
%     'mu_km3s2'        [398600]
%     'Epoch'           [datetime(2026,1,1)]  time of out.t = 0
%     'SunExcl_deg'     [60]      Sun exclusion angle
%     'MoonExcl_deg'    [30]      Moon exclusion angle
%     'Tint_s'          []        only needed if sol lacks scienceStartNode/info
%     'Wavelength_m'    []        if given, u-v axes are in M-lambda, else meters
%     'SkyFile'         ['']      .mat file saved by Sky_Coverage_Runner
%     'MakeFigures'     [true]
%     'UseEphemeris'    [true]    Sun/Moon angles and eclipse (Aerospace Toolbox)
%     'FormationCase'   [1]       case used for the formation figure
%     'FontName' ['Times New Roman'], 'FontSize' [9], 'Resolution' [600]
%     'SingleColumnIn' [3.3125], 'DoubleColumnIn' [6.75], 'Visible' ['off']
%
%   OUTPUT
%     R.cases, R.tables (MATLAB tables), R.files, R.notes, R.formation, R.sky
%     Writes .csv and .tex tables to <OutDir>/tables and .pdf/.png to <OutDir>/figures.

%% ------------------------------------------------------------------ options
p = inputParser;
p.FunctionName = 'generate_paper_results';
p.addParameter('OutDir', 'paper_results');
p.addParameter('Labels', {});
p.addParameter('TargetNames', {});
p.addParameter('Dmax_m', 5);
p.addParameter('ConstraintTol_m', 0.1);
p.addParameter('T_N', 0.04);
p.addParameter('Isp_s', 1500);
p.addParameter('RE_km', 6378);
p.addParameter('mu_km3s2', 398600);
p.addParameter('Epoch', datetime(2026,1,1,0,0,0));
p.addParameter('SunExcl_deg', 60);
p.addParameter('MoonExcl_deg', 30);
p.addParameter('Tint_s', []);
p.addParameter('Wavelength_m', []);
p.addParameter('SkyFile', '');
p.addParameter('MakeFigures', true);
p.addParameter('UseEphemeris', true);
p.addParameter('FormationCase', 1);
p.addParameter('FontName', 'Times New Roman');
p.addParameter('FontSize', 9);
p.addParameter('Resolution', 600);
p.addParameter('SingleColumnIn', 3.3125);
p.addParameter('DoubleColumnIn', 6.75);
p.addParameter('Visible', 'off');
p.parse(varargin{:});
o = p.Results;
o.Labels = cellstr(o.Labels);
o.TargetNames = cellstr(o.TargetNames);

%% ------------------------------------------------------------ normalize inputs
if isstruct(out), out = {out}; end
if isstruct(sol), sol = {sol}; end
nCase = numel(sol);
if numel(out) == 1 && nCase > 1, out = repmat(out, 1, nCase); end
if numel(out) ~= nCase
    error('out and sol must have the same number of cases (or a single out).');
end

outDir = char(o.OutDir);
figDir = fullfile(outDir, 'figures');
texDir = fullfile(outDir, 'tables');
for d = {outDir, figDir, texDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

R = struct();
R.files = struct('tables', {{}}, 'figures', {{}});
R.notes = {};

%% ------------------------------------------------------------- case analysis
cases = cell(1, nCase);
for i = 1:nCase
    cases{i} = analyze_case(out{i}, sol{i}, o, i);
    for q = 1:numel(cases{i}.notes)
        R.notes{end+1} = sprintf('[%s] %s', cases{i}.m.label, cases{i}.notes{q}); %#ok<AGROW>
    end
end
R.cases = cases;

%% ------------------------------------------------------------------- tables
% Targets (tab:targets)
[T, hdr, rows] = build_targets(cases);
R.tables.targets = T;
write_tex_table(fullfile(texDir,'tab_targets.tex'), ...
    'Science targets and their directions in the combiner RTN frame at $u_c=0$.', ...
    'tab:targets', '|l|c|c|c|c|', hdr, rows, '\small');
R = save_table(R, T, texDir, 'tab_targets');

% Optimal control summary (tab:ocpres)
[T, hdr, rows] = build_ocp(cases);
R.tables.ocp = T;
write_tex_table(fullfile(texDir,'tab_ocpres.tex'), ...
    'Summary of the optimal control solutions.', 'tab:ocpres', ...
    '|l|c|c|c|c|c|c|', hdr, rows, '\small');
R = save_table(R, T, texDir, 'tab_ocpres');

% Extended per-case metrics (CSV only)
R.tables.extended = build_extended(cases);
R = save_table(R, R.tables.extended, texDir, 'tab_extended');

% Hold timing (tab:hold) and keep-out/eclipse (tab:keepout)
[T, hdr, rows] = build_hold(cases);
R.tables.hold = T;
write_tex_table(fullfile(texDir,'tab_hold.tex'), ...
    'Handoff, settling, and OPD spread of the uncontrolled and controlled formation.', ...
    'tab:hold', '|l|c|c|c|c|c|c|c|', hdr, rows, '\footnotesize');
R = save_table(R, T, texDir, 'tab_hold');

[T, hdr, rows] = build_keepout(cases, o);
R.tables.keepout = T;
write_tex_table(fullfile(texDir,'tab_keepout.tex'), ...
    'Angles from the target to the Sun and Moon at the start of science, and eclipse time.', ...
    'tab:keepout', '|l|c|c|c|c|c|', hdr, rows, '\small');
R = save_table(R, T, texDir, 'tab_keepout');

% Differential perturbations (tab:diffpert) from the first out that has the report
iP = find(cellfun(@(x) isfield(x,'perturbations') && isfield(x.perturbations,'peaks'), out), 1);
if ~isempty(iP)
    [T, hdr, rows, M] = build_diffpert(out{iP}.perturbations);
    R.tables.diffpert = T;
    R.diffpert.matrix_nmps2 = M;
    if isfield(out{iP}.perturbations, 'SRPtoNextLargest')
        R.diffpert.SRPtoNextLargest = out{iP}.perturbations.SRPtoNextLargest;
    end
    write_tex_table(fullfile(texDir,'tab_diffpert.tex'), ...
        'Peak differential acceleration between each collector and the combiner [nm/s$^2$].', ...
        'tab:diffpert', '|l|c|c|c|', hdr, rows, '\small');
    R = save_table(R, T, texDir, 'tab_diffpert');
else
    R.notes{end+1} = 'out.perturbations not found: differential-perturbation table skipped.';
end

% u-v coverage (tab:uv)
[T, hdr, rows] = build_uv(cases, o);
R.tables.uv = T;
write_tex_table(fullfile(texDir,'tab_uv.tex'), ...
    'Projected baseline range and $u$--$v$ coverage area during the science interval.', ...
    'tab:uv', '|l|c|c|c|c|', hdr, rows, '\small');
R = save_table(R, T, texDir, 'tab_uv');

% Sky coverage (tab:sky)
S = [];
if ~isempty(o.SkyFile)
    S = load_sky(o.SkyFile);
    [T, hdr, rows] = build_sky(S);
    R.tables.sky = T;
    R.sky = rmfield(S, {'RA','DEC','cov'});
    write_tex_table(fullfile(texDir,'tab_sky.tex'), ...
        'Sky coverage statistics for 16,290 targets over one year (hourly samples).', ...
        'tab:sky', '|l|c|', hdr, rows, '\small');
    R = save_table(R, T, texDir, 'tab_sky');
end

%% ------------------------------------------------------------------ figures
if o.MakeFigures
    fo = o.FormationCase;
    if fo >= 1 && fo <= nCase
        [files, R.formation] = fig_formation(out{fo}, cases{fo}, o, figDir);
        R.files.figures = [R.files.figures, files];
    end
    R.files.figures = [R.files.figures, fig_uncontrolled(cases, o, figDir)];
    if ~isempty(iP)
        R.files.figures = [R.files.figures, fig_diffpert(out{iP}.perturbations, o, figDir)];
    end
    for i = 1:nCase
        tag = sanitize(cases{i}.m.label);
        R.files.figures = [R.files.figures, fig_ocp_opd(cases{i}, o, figDir, tag)];
        R.files.figures = [R.files.figures, fig_ocp_controls(cases{i}, o, figDir, tag)];
        R.files.figures = [R.files.figures, fig_uv(cases{i}, o, figDir, tag)];
    end
    if nCase > 1
        R.files.figures = [R.files.figures, fig_uv_all(cases, o, figDir)];
    end
    if ~isempty(S)
        R.files.figures = [R.files.figures, fig_sky(S, cases, o, figDir)];
    end
end

%% ------------------------------------------------------------------- report
save(fullfile(outDir, 'paper_results.mat'), 'R');
fprintf('\n=== generate_paper_results: %d case(s) ===\n', nCase);
disp(R.tables.ocp);
disp(R.tables.hold);
if ~isempty(R.notes)
    fprintf('\nNOTES AND CHECKS (read these):\n');
    for q = 1:numel(R.notes), fprintf('  - %s\n', R.notes{q}); end
else
    fprintf('\nAll consistency checks passed.\n');
end
fprintf('\nTables: %s\nFigures: %s\n', texDir, figDir);
end

%% =========================================================================
%                              CASE ANALYSIS
%% =========================================================================
function c = analyze_case(out, sol, o, iCase)
req = {'t','chief','Ncollectors','X','Alpha','Dir','RTN','OPD_km','OPDext_km','relay_km','massUsed_kg','Sopd'};
for ii = 1:numel(req)
    if ~isfield(sol, req{ii})
        error('sol (case %d) is missing field "%s".', iCase, req{ii});
    end
end
notes = {};
g0 = 9.80665;
Dmax = o.Dmax_m;

t  = sol.t(:);
Nt = numel(t);
Nc = sol.Ncollectors;
h  = t(2) - t(1);
dt = diff(t);

% ---- science start node and interval
k_s = find_science_node(sol, o, Nt, h, iCase);
if k_s < 2 || k_s >= Nt
    error('Invalid science start node %d (Nt = %d) in case %d.', k_s, Nt, iCase);
end
idxS = (k_s:Nt).';
Tint_actual_s = t(Nt) - t(k_s);
Tint_nom_h = round(Tint_actual_s/600)/6;

% ---- target
[ra, dec, phi0, beta] = get_star(out);
tgt = lookup_target(ra, dec, o.TargetNames, iCase);
if numel(o.Labels) >= iCase && ~isempty(o.Labels{iCase})
    label = o.Labels{iCase};
else
    label = sprintf('%s, %g h', tgt, Tint_nom_h);
end

% ---- OPD (complete relayed-beam path), external and relay terms
OPD_m   = 1000*sol.OPD_km;
spread_m = max(OPD_m,[],2) - min(OPD_m,[],2);
relOPD_m = OPD_m - mean(OPD_m, 2);
ext_m   = 1000*sol.OPDext_km;
relay_m = 1000*sol.relay_km;
extSpread_m   = max(ext_m,[],2)   - min(ext_m,[],2);
relaySpread_m = max(relay_m,[],2) - min(relay_m,[],2);

maxSpreadSci_m = max(spread_m(idxS));
viol_m = max(0, maxSpreadSci_m - Dmax);
boundMet = viol_m <= o.ConstraintTol_m;
if ~boundMet
    notes{end+1} = sprintf('OPD bound violated in science interval by %.3f m (tolerance %.3f m).', viol_m, o.ConstraintTol_m);
end

% ---- slack
Sopd = sol.Sopd;
if size(Sopd,1) ~= Nc && size(Sopd,2) == Nc, Sopd = Sopd.'; end
slack_m = 1000*Sopd;                                  % Nc x Nt
slackSettle_m = max([0; reshape(slack_m(:,1:k_s-1), [], 1)]);
slackSci_m    = max([0; reshape(slack_m(:,k_s:end), [], 1)]);
slackMax_m    = max(slack_m(:));

% ---- time to enter the limit (stays inside afterwards)
ok = spread_m <= Dmax + o.ConstraintTol_m;
kBad = find(~ok, 1, 'last');
if isempty(kBad),      tEnter_s = t(1);
elseif kBad == Nt,     tEnter_s = NaN;
else,                  tEnter_s = t(kBad+1);
end

% ---- geometry: ranges and pair separations
range_km = zeros(Nt, Nc);
for j = 1:Nc, range_km(:,j) = sqrt(sum(sol.RTN{j}.^2, 2)); end
pairs = nchoosek(1:Nc, 2);
sep_km = zeros(Nt, size(pairs,1));
for q = 1:size(pairs,1)
    sep_km(:,q) = sqrt(sum((sol.RTN{pairs(q,1)} - sol.RTN{pairs(q,2)}).^2, 2));
end

% ---- controls, mass, delta-v
alpha = zeros(Nt-1, Nc);
dirv  = zeros(Nt-1, 3, Nc);
massUsedT = zeros(Nt, Nc);
dvCum = zeros(Nt, Nc);
dvOwn = zeros(1, Nc);
massPred = zeros(1, Nc);
for j = 1:Nc
    a = sol.Alpha{j}(:);
    D = sol.Dir{j};
    if size(D,1) == 3 && size(D,2) ~= 3, D = D.'; end
    m = sol.X{j}(:,7);
    alpha(:,j)  = a;
    dirv(:,:,j) = D;
    mmid = 0.5*(m(1:end-1) + m(2:end));
    acc  = (o.T_N ./ mmid) .* a;
    dvCum(2:end,j) = cumsum(acc .* dt);
    dvOwn(j) = dvCum(end,j);
    massPred(j) = sum(o.T_N * a .* dt) / (o.Isp_s * g0);
    massUsedT(:,j) = m(1) - m;
end
scale = ones(1, Nc);
if isfield(sol, 'dvApprox_mps') && numel(sol.dvApprox_mps) == Nc
    scale = sol.dvApprox_mps(:).' ./ dvOwn;
    if max(abs(scale - 1)) > 1e-6
        notes{end+1} = sprintf(['T_N option (%.4g N) differs from the thrust used in the OCP ', ...
            '(implied %.4g N). Delta-v taken from sol.dvApprox_mps. Set ''T_N'' correctly.'], ...
            o.T_N, o.T_N*mean(scale));
    end
end
dvCum = dvCum .* scale;
dv = dvOwn .* scale;
massUsed = sol.massUsed_kg(:).';
mrel = max(abs(massPred.*scale - massUsed) ./ max(massUsed, eps));
if mrel > 1e-3
    notes{end+1} = sprintf(['Propellant from sol.massUsed_kg disagrees with Isp = %.0f s by %.2g%% ', ...
        '(set ''Isp_s'' to the value used in the OCP).'], o.Isp_s, 100*mrel);
end
m0v = cellfun(@(x) x(1,7),   sol.X);
mfv = cellfun(@(x) x(end,7), sol.X);
dvRocket = o.Isp_s * g0 * log(m0v ./ mfv);

% ---- handoff match in the uncontrolled propagation and comparison
[du, k0] = min(abs(out.states.chief(:,6) - sol.chief(1,6)));
if du > 1e-6
    notes{end+1} = sprintf('Handoff match: chief u differs by %.2e rad. Check that out and sol belong together.', du);
end
tOut = out.t(:);
t0_s = tOut(k0);
if isfield(out, 'OPDfullSpread_m')
    uncAll = out.OPDfullSpread_m(:);
else
    uncAll = 1000*(max(out.OPDfull_km,[],2) - min(out.OPDfull_km,[],2));
end
unc_m = interp1(tOut - t0_s, uncAll, t, 'linear', NaN);

% ---- keep-out geometry and eclipse
sun_deg = NaN; moon_deg = NaN; eclSci_min = NaN; eclHold_min = NaN;
if o.UseEphemeris
    try
        jd0 = juliandate(o.Epoch);
        sI = [cos(dec)*cos(ra); cos(dec)*sin(ra); sin(dec)];
        jdSci = jd0 + (t0_s + t(k_s))/86400;
        rS = planetEphemeris(jdSci, 'Earth', 'Sun',  '421'); rS = rS(:);
        rM = planetEphemeris(jdSci, 'Earth', 'Moon', '421'); rM = rM(:);
        sun_deg  = acosd(max(-1, min(1, dot(sI, rS/norm(rS)))));
        moon_deg = acosd(max(-1, min(1, dot(sI, rM/norm(rM)))));
        jdAll = jd0 + (t0_s + t)/86400;
        rSall = planetEphemeris(jdAll, 'Earth', 'Sun', '421');
        if size(rSall,2) ~= 3, rSall = rSall.'; end
        ecl = shadow_flags(sol.chief, rSall, o.RE_km);
        eclHold_min = sum(ecl(1:end-1)) * h / 60;
        eclSci_min  = sum(ecl(k_s:end-1)) * h / 60;
        if sun_deg < o.SunExcl_deg
            notes{end+1} = sprintf('Target is %.1f deg from the Sun at science start (inside the %g deg keep-out).', sun_deg, o.SunExcl_deg);
        end
        if moon_deg < o.MoonExcl_deg
            notes{end+1} = sprintf('Target is %.1f deg from the Moon (geocentric) at science start (inside the %g deg keep-out).', moon_deg, o.MoonExcl_deg);
        end
        if eclSci_min > 0
            notes{end+1} = sprintf('Combiner is in eclipse for %.1f min of the science interval.', eclSci_min);
        end
    catch ME
        notes{end+1} = ['Ephemeris-based checks skipped: ' ME.message];
    end
end

% ---- u-v coverage (collector-to-collector baselines, science interval)
uvr = compute_uv(sol, ra, dec, idxS, o.Wavelength_m);
if uvr.wCheck_m > 1e-6
    notes{end+1} = sprintf('u-v rotation check: line-of-sight component differs from OCP OPD terms by %.2e m.', uvr.wCheck_m);
end

% ---- assemble
m = struct();
m.label = label; m.target = tgt; m.ra = ra; m.dec = dec; m.phi0 = phi0; m.beta = beta;
m.Tint_h = Tint_nom_h; m.Tint_actual_s = Tint_actual_s; m.k_s = k_s; m.Nt = Nt; m.h_s = h;
m.success  = isfield(sol,'success') && logical(sol.success);
m.success1 = isfield(sol,'phase1') && isfield(sol.phase1,'success') && logical(sol.phase1.success);
m.dv = dv;
m.dvMean = mean(dv); m.dvMin = min(dv); m.dvMax = max(dv);
m.dvRocketMean = mean(dvRocket);
m.propTotal = sum(massUsed);
m.propMean  = mean(massUsed);
m.slackMax_m = slackMax_m; m.slackSettle_m = slackSettle_m; m.slackSci_m = slackSci_m;
m.rangeMin_km = min(range_km(:)); m.rangeMax_km = max(range_km(:));
m.minSep_km = min(sep_km(:));
m.maxSpreadSci_m = maxSpreadSci_m; m.viol_m = viol_m; m.boundMet = boundMet;
m.tEnter_min = tEnter_s/60;
m.t0_s = t0_s; m.settle_min = t(k_s)/60;
m.uncSpread0_m = unc_m(1); m.uncSpreadSci0_m = unc_m(k_s); m.uncMaxSci_m = max(unc_m(idxS));
m.ctrlSpread0_m = spread_m(1); m.ctrlSpreadSci0_m = spread_m(k_s);
m.alphaMean = mean(alpha(:)); m.alphaMin = min(alpha(:)); m.alphaMax = max(alpha(:));
m.alphaSciMean = mean(reshape(alpha(k_s:end,:), [], 1));
m.sun_deg = sun_deg; m.moon_deg = moon_deg; m.eclSci_min = eclSci_min; m.eclHold_min = eclHold_min;

d = struct();
d.t_s = t; d.k_s = k_s; d.spread_m = spread_m; d.unc_m = unc_m; d.relOPD_m = relOPD_m;
d.extSpread_m = extSpread_m; d.relaySpread_m = relaySpread_m; d.range_km = range_km;
d.alpha = alpha; d.dir = dirv; d.massUsedT = massUsedT; d.dvCum = dvCum; d.slack_m = slack_m;
d.outT_s = tOut - t0_s; d.outSpread_m = uncAll; d.outT_abs_s = tOut;

c = struct('m', m, 'd', d, 'uv', uvr, 'notes', {notes});
end

function k_s = find_science_node(sol, o, Nt, h, iCase)
if isfield(sol,'scienceStartNode') && ~isempty(sol.scienceStartNode)
    k_s = sol.scienceStartNode;
elseif isfield(sol,'info') && isstruct(sol.info) && isfield(sol.info,'settleDuration_s')
    k_s = round(sol.info.settleDuration_s/h) + 1;
elseif ~isempty(o.Tint_s)
    Tint = o.Tint_s(min(iCase, numel(o.Tint_s)));
    k_s = Nt - ceil(Tint/h);
else
    error(['Cannot locate the science start (case %d). Save sol.scienceStartNode ', ...
           'or sol.info, or pass ''Tint_s''.'], iCase);
end
end

function [ra, dec, phi0, beta] = get_star(out)
if isfield(out,'star') && all(isfield(out.star, {'ra','dec','phi0','beta'}))
    ra = out.star.ra; dec = out.star.dec; phi0 = out.star.phi0; beta = out.star.beta;
else
    error('out.star must contain ra, dec, phi0 and beta (set in Example_GEO_Formation_OPD).');
end
end

function name = lookup_target(ra, dec, userNames, idx)
if numel(userNames) >= idx && ~isempty(userNames{idx}), name = userNames{idx}; return; end
known = {'Altair', 5.1832, 0.1548; 'Deneb', 5.403, 0.800; ...
         'Beta Pictoris', 1.514, -0.892; 'Tau Ceti', 0.4600, -0.2781};
name = sprintf('RA %.1f, Dec %.1f deg', rad2deg(ra), rad2deg(dec));
for k = 1:size(known,1)
    if abs(ra - known{k,2}) < 1e-3 && abs(dec - known{k,3}) < 1e-3
        name = known{k,1}; return;
    end
end
end

function ecl = shadow_flags(chiefQNS, rSun_km, RE)
% Same shadow rule as the OCP dynamics: in shadow if theta_sat + theta_sun <= theta.
a = chiefQNS(:,1); inc = chiefQNS(:,4); O = chiefQNS(:,5); u = chiefQNS(:,6);
Rhat = [cos(O).*cos(u) - sin(O).*sin(u).*cos(inc), ...
        sin(O).*cos(u) + cos(O).*sin(u).*cos(inc), ...
        sin(u).*sin(inc)];
rs = a .* Rhat;
nr = sqrt(sum(rs.^2, 2));
ns = sqrt(sum(rSun_km.^2, 2));
th   = acos(max(-1, min(1, sum(rs.*rSun_km, 2) ./ (nr.*ns))));
thSt = acos(min(1, RE ./ nr));
thSn = acos(min(1, RE ./ ns));
ecl = (thSt + thSn) <= th;
end

function uvr = compute_uv(sol, ra, dec, idx, lambda_m)
% u-v points = baselines between PAIRS OF COLLECTORS projected on the plane normal
% to the (inertially fixed) star direction. East = u, north = v.
Nc = sol.Ncollectors; ns = numel(idx);
sI = [cos(dec)*cos(ra); cos(dec)*sin(ra); sin(dec)];
uHat = cross([0;0;1], sI);
if norm(uHat) < 1e-9, uHat = [1;0;0]; end
uHat = uHat/norm(uHat);
vHat = cross(sI, uHat);
rI = repmat({zeros(ns,3)}, 1, Nc);
for n = 1:ns
    k = idx(n);
    Q = rtn_to_eci(sol.chief(k,4), sol.chief(k,5), sol.chief(k,6));
    for j = 1:Nc
        rI{j}(n,:) = (1000 * Q * sol.RTN{j}(k,:).').';      % meters
    end
end
pairs = nchoosek(1:Nc, 2); Np = size(pairs,1);
uvm = zeros(ns, 2, Np); w = zeros(ns, Np); len3 = zeros(ns, Np);
ext = 1000*sol.OPDext_km(idx,:);
wCheck = 0;
for q = 1:Np
    b = rI{pairs(q,2)} - rI{pairs(q,1)};
    uvm(:,1,q) = b*uHat;
    uvm(:,2,q) = b*vHat;
    w(:,q)    = b*sI;
    len3(:,q) = sqrt(sum(b.^2, 2));
    wCheck = max(wCheck, max(abs(w(:,q) - (ext(:,pairs(q,2)) - ext(:,pairs(q,1))))));
end
proj = zeros(ns, Np);
for q = 1:Np, proj(:,q) = sqrt(uvm(:,1,q).^2 + uvm(:,2,q).^2); end
pts = reshape(permute(uvm, [1 3 2]), [], 2);
pts = unique([pts; -pts], 'rows');
area = 0;
if size(pts,1) >= 3
    try
        kh = convhull(pts(:,1), pts(:,2));
        area = polyarea(pts(kh,1), pts(kh,2));
    catch
        area = 0;
    end
end
if isempty(lambda_m)
    scale = 1; unit = 'm';
else
    scale = 1/(lambda_m*1e6); unit = 'M\lambda';
end
uvr = struct('uv_m', uvm, 'pairs', pairs, 'w_m', w, 'len3D_m', len3, ...
    'projMin_m', min(proj(:)), 'projMax_m', max(proj(:)), 'len3DMin_m', min(len3(:)), ...
    'len3DMax_m', max(len3(:)), 'hull_m2', area, 'nSamples', ns, ...
    'wCheck_m', wCheck, 'scale', scale, 'unit', unit);
end

function Q = rtn_to_eci(inc, O, u)
cO = cos(O); sO = sin(O); ci = cos(inc); si = sin(inc); cu = cos(u); su = sin(u);
Q = [cO*cu - sO*su*ci, -cO*su - sO*cu*ci,  sO*si; ...
     sO*cu + cO*su*ci, -sO*su + cO*cu*ci, -cO*si; ...
     su*si,             cu*si,             ci];
end

%% =========================================================================
%                                TABLES
%% =========================================================================
function [T, hdr, rows] = build_targets(cases)
seen = zeros(0,2); nm = {}; ra = []; dec = []; ph = []; be = [];
for i = 1:numel(cases)
    m = cases{i}.m; key = [m.ra, m.dec];
    if any(all(abs(seen - key) < 1e-9, 2)), continue; end
    seen(end+1,:) = key; %#ok<AGROW>
    nm{end+1,1} = m.target; ra(end+1,1) = m.ra; dec(end+1,1) = m.dec; %#ok<AGROW>
    ph(end+1,1) = rad2deg(m.phi0); be(end+1,1) = rad2deg(m.beta); %#ok<AGROW>
end
T = table(nm, ra, rad2deg(ra), dec, rad2deg(dec), ph, be, 'VariableNames', ...
    {'Target','RA_rad','RA_deg','Dec_rad','Dec_deg','Phi0_deg','Beta_deg'});
hdr = {'Target','$\alpha$ [rad (deg)]','$\delta$ [rad (deg)]','$\phi_0$ [deg]','$\beta$ [deg]'};
rows = cell(numel(nm), 5);
for i = 1:numel(nm)
    rows(i,:) = {nm{i}, ...
        [num2tex(ra(i),'%.4f') ' (' num2tex(rad2deg(ra(i)),'%.2f') ')'], ...
        [num2tex(dec(i),'%.4f') ' (' num2tex(rad2deg(dec(i)),'%.2f') ')'], ...
        num2tex(ph(i),'%.1f'), num2tex(be(i),'%.1f')};
end
end

function [T, hdr, rows] = build_ocp(cases)
n = numel(cases);
nm = cell(n,1); Tint = zeros(n,1); dvM = zeros(n,1); dvLo = dvM; dvHi = dvM;
prop = dvM; slack = dvM; rLo = dvM; rHi = dvM; conv = false(n,1);
rows = cell(n, 7);
for i = 1:n
    m = cases{i}.m;
    nm{i} = m.target; Tint(i) = m.Tint_h; dvM(i) = m.dvMean; dvLo(i) = m.dvMin; dvHi(i) = m.dvMax;
    prop(i) = m.propTotal; slack(i) = m.slackMax_m; rLo(i) = m.rangeMin_km; rHi(i) = m.rangeMax_km;
    conv(i) = m.success;
    rows(i,:) = {m.target, num2tex(m.Tint_h,'%g'), ...
        [num2tex(m.dvMean,'%.2f') ' (' num2tex(m.dvMin,'%.2f') '--' num2tex(m.dvMax,'%.2f') ')'], ...
        num2tex(m.propTotal,'%.2f'), num2tex(m.slackMax_m,'%.3g'), ...
        [num2tex(m.rangeMin_km,'%.3f') '--' num2tex(m.rangeMax_km,'%.3f')], ...
        yesno(m.success)};
end
T = table(nm, Tint, dvM, dvLo, dvHi, prop, slack, rLo, rHi, conv, 'VariableNames', ...
    {'Target','Tint_h','dv_mean_mps','dv_min_mps','dv_max_mps','Propellant_total_kg', ...
     'MaxSlack_m','Range_min_km','Range_max_km','Converged'});
hdr = {'Target','$T_{\mathrm{int}}$ [h]','$\Delta v$ per collector [m/s]', ...
       'Propellant, total [kg]','Maximum slack [m]','$\ell_j$ range [km]','Converged'};
end

function T = build_extended(cases)
rowsS = cell(numel(cases), 1);
for i = 1:numel(cases)
    m = cases{i}.m;
    s = rmfield(m, 'dv');
    for j = 1:numel(m.dv), s.(sprintf('dv%d_mps', j)) = m.dv(j); end
    rowsS{i} = s;
end
T = struct2table([rowsS{:}]');
end

function [T, hdr, rows] = build_hold(cases)
n = numel(cases);
nm = cell(n,1); Tint = zeros(n,1); t0 = Tint; settle = Tint; u0 = Tint; us = Tint;
tent = Tint; mx = Tint; viol = Tint;
rows = cell(n, 8);
for i = 1:n
    m = cases{i}.m;
    nm{i} = m.target; Tint(i) = m.Tint_h; t0(i) = m.t0_s/3600; settle(i) = m.settle_min;
    u0(i) = m.uncSpread0_m; us(i) = m.uncSpreadSci0_m; tent(i) = m.tEnter_min;
    mx(i) = m.maxSpreadSci_m; viol(i) = m.viol_m;
    rows(i,:) = {m.target, num2tex(m.Tint_h,'%g'), num2tex(m.t0_s/3600,'%.2f'), ...
        num2tex(m.settle_min,'%.1f'), num2tex(m.uncSpread0_m,'%.3g'), ...
        num2tex(m.uncSpreadSci0_m,'%.3g'), num2tex(m.tEnter_min,'%.1f'), ...
        num2tex(m.maxSpreadSci_m,'%.3f')};
end
T = table(nm, Tint, t0, settle, u0, us, tent, mx, viol, 'VariableNames', ...
    {'Target','Tint_h','Handoff_h','Settling_min','UncontrolledSpreadHandoff_m', ...
     'UncontrolledSpreadScienceStart_m','TimeToLimit_min','MaxSpreadScience_m','Violation_m'});
hdr = {'Target','$T_{\mathrm{int}}$ [h]','Handoff [h]','Settling [min]', ...
       'Uncontrolled spread, handoff [m]','Uncontrolled spread, science start [m]', ...
       'Time to reach limit [min]','Maximum spread, science [m]'};
end

function [T, hdr, rows] = build_keepout(cases, o)
n = numel(cases);
nm = cell(n,1); Tint = zeros(n,1); sun = Tint; moon = Tint; es = Tint; eh = Tint;
rows = cell(n, 6);
for i = 1:n
    m = cases{i}.m;
    nm{i} = m.target; Tint(i) = m.Tint_h; sun(i) = m.sun_deg; moon(i) = m.moon_deg;
    es(i) = m.eclSci_min; eh(i) = m.eclHold_min;
    rows(i,:) = {m.target, num2tex(m.Tint_h,'%g'), num2tex(m.sun_deg,'%.1f'), ...
        num2tex(m.moon_deg,'%.1f'), num2tex(m.eclHold_min,'%.1f'), num2tex(m.eclSci_min,'%.1f')};
end
T = table(nm, Tint, sun, moon, eh, es, 'VariableNames', ...
    {'Target','Tint_h','SunAngle_deg','MoonAngle_deg_geocentric','EclipseHold_min','EclipseScience_min'});
hdr = {'Target','$T_{\mathrm{int}}$ [h]', ...
       sprintf('Sun angle [deg] (limit %g)', o.SunExcl_deg), ...
       sprintf('Moon angle [deg] (limit %g)', o.MoonExcl_deg), ...
       'Eclipse, hold [min]','Eclipse, science [min]'};
end

function [T, hdr, rows, M] = build_diffpert(rep)
pk = rep.peaks;
names = {'J2','SRP','Moon','Sun'};
disp_names = {'Earth $J_2$','SRP','Moon','Sun'};
Nc = max(pk.Collector);
M = nan(4, Nc);
src = cellstr(pk.Source);
for s = 1:4
    for j = 1:Nc
        idx = strcmp(src, names{s}) & (pk.Collector == j);
        if any(idx), M(s,j) = pk.PeakDifferential_nmps2(find(idx,1)); end
    end
end
vn = arrayfun(@(j) sprintf('Collector%d_nmps2', j), 1:Nc, 'UniformOutput', false);
T = array2table(M, 'VariableNames', vn);
T = addvars(T, names(:), 'Before', 1, 'NewVariableNames', 'Source');
hdr = [{'Source'}, arrayfun(@(j) sprintf('Collector %d', j), 1:Nc, 'UniformOutput', false)];
rows = cell(4, Nc+1);
for s = 1:4
    rows{s,1} = disp_names{s};
    for j = 1:Nc, rows{s,j+1} = num2tex(M(s,j), '%.3g'); end
end
end

function [T, hdr, rows] = build_uv(cases, o)
n = numel(cases);
nm = cell(n,1); Tint = zeros(n,1); lo = Tint; hi = Tint; ar = Tint; ns = Tint;
lo3 = Tint; hi3 = Tint;
rows = cell(n, 5);
for i = 1:n
    m = cases{i}.m; u = cases{i}.uv;
    nm{i} = m.target; Tint(i) = m.Tint_h; lo(i) = u.projMin_m; hi(i) = u.projMax_m;
    ar(i) = u.hull_m2; ns(i) = u.nSamples; lo3(i) = u.len3DMin_m; hi3(i) = u.len3DMax_m;
    rows(i,:) = {m.target, num2tex(m.Tint_h,'%g'), ...
        [num2tex(u.projMin_m,'%.0f') '--' num2tex(u.projMax_m,'%.0f')], ...
        num2tex(u.hull_m2,'%.3g'), sprintf('%d', u.nSamples)};
end
T = table(nm, Tint, lo, hi, lo3, hi3, ar, ns, 'VariableNames', ...
    {'Target','Tint_h','ProjBaselineMin_m','ProjBaselineMax_m','Baseline3DMin_m', ...
     'Baseline3DMax_m','HullArea_m2','Samples'});
hdr = {'Target','$T_{\mathrm{int}}$ [h]','Shortest--longest projected baseline [m]', ...
       'Convex hull area [m$^2$]','Samples'};
if ~isempty(o.Wavelength_m)
    T.ProjBaselineMin_Mlambda = lo/(o.Wavelength_m*1e6);
    T.ProjBaselineMax_Mlambda = hi/(o.Wavelength_m*1e6);
end
end

function S = load_sky(file)
vars = {'metrics','RA_mesh','DEC_mesh','max_continuous','revisit_times', ...
        'time_analysis','sc_positions_ECI','params'};
info = whos('-file', file);
keep = vars(ismember(vars, {info.name}));
D = load(file, keep{:});
need = {'metrics','RA_mesh','DEC_mesh','max_continuous','revisit_times','time_analysis'};
for k = 1:numel(need)
    if ~isfield(D, need{k}), error('Sky file is missing variable "%s".', need{k}); end
end
cov = D.metrics.coverage_percent(:);
dt_s = seconds(D.time_analysis(2) - D.time_analysis(1));
mc_h = D.max_continuous(:)/3600;
guar_h = max(mc_h - dt_s/3600, 0);
wins = [2 3 4 6];
pctW = arrayfun(@(w) 100*mean(mc_h >= w + dt_s/3600), wins);
gaps = cell2mat(cellfun(@(x) x(:), D.revisit_times(:), 'UniformOutput', false)) / 3600;
S = struct();
S.RA = D.RA_mesh(:); S.DEC = D.DEC_mesh(:); S.cov = cov;
S.nTargets = numel(cov);
S.nSamples = numel(D.time_analysis);
S.dt_s = dt_s;
S.totalAccessible_pct = 100*mean(cov > 0);
S.minCoverage_pct = min(cov);
S.meanCoverage_pct = mean(cov);
S.medianCoverage_pct = median(cov);
S.medianGuaranteedWindow_h = median(guar_h);
S.maxGuaranteedWindow_h = max(guar_h);
S.windowHours = wins;
S.pctTargetsWithWindow = pctW;
S.medianRevisit_h = median(gaps);
S.meanRevisit_h = mean(gaps);
S.eclipse_pct = NaN;
if isfield(D,'sc_positions_ECI') && isfield(D,'params') && isfield(D.params,'ephem')
    r = D.sc_positions_ECI;
    sunv = D.params.ephem.rSun - r;
    earthv = -r;
    nS = sqrt(sum(sunv.^2, 2)); nE = sqrt(sum(earthv.^2, 2));
    cs = sum(sunv.*earthv, 2) ./ (nS.*nE);
    ang = acosd(max(-1, min(1, cs)));
    angR = asind(min(1, D.params.const.R_earth_km ./ nE));
    S.eclipse_pct = 100*mean(sum(sunv.*earthv, 2) > 0 & ang < angR);
end
end

function [T, hdr, rows] = build_sky(S)
q = {'Sky accessible at least once [%]'; 'Minimum coverage [% of time]'; ...
     'Mean coverage [% of time]'; 'Median coverage [% of time]'; ...
     'Combiner in eclipse [% of time]'; 'Median longest guaranteed window [h]'; ...
     'Maximum longest guaranteed window [h]'; ...
     'Targets with guaranteed window >= 2 h [%]'; '... >= 3 h [%]'; ...
     '... >= 4 h [%]'; '... >= 6 h [%]'; 'Median revisit time [h]'; 'Mean revisit time [h]'};
v = [S.totalAccessible_pct; S.minCoverage_pct; S.meanCoverage_pct; S.medianCoverage_pct; ...
     S.eclipse_pct; S.medianGuaranteedWindow_h; S.maxGuaranteedWindow_h; ...
     S.pctTargetsWithWindow(:); S.medianRevisit_h; S.meanRevisit_h];
T = table(q, v, 'VariableNames', {'Quantity','Value'});
hdr = {'Quantity','Value'};
rows = {'Sky accessible at least once [\%]', num2tex(S.totalAccessible_pct,'%.2f'); ...
    'Minimum coverage [\% of time]', num2tex(S.minCoverage_pct,'%.1f'); ...
    'Mean, median coverage [\% of time]', [num2tex(S.meanCoverage_pct,'%.1f') ', ' num2tex(S.medianCoverage_pct,'%.1f')]; ...
    'Combiner in eclipse [\% of time]', num2tex(S.eclipse_pct,'%.2f'); ...
    'Median, maximum of the longest guaranteed window [h]', [num2tex(S.medianGuaranteedWindow_h,'%.0f') ', ' num2tex(S.maxGuaranteedWindow_h,'%.0f')]; ...
    'Targets with a guaranteed window of at least 2, 3, 4, 6~h [\%]', strjoin(arrayfun(@(x) num2tex(x,'%.1f'), S.pctTargetsWithWindow, 'UniformOutput', false), ', '); ...
    'Median, mean revisit time [h]', [num2tex(S.medianRevisit_h,'%.1f') ', ' num2tex(S.meanRevisit_h,'%.1f')]};
end

function R = save_table(R, T, texDir, base)
f = fullfile(texDir, [base '.csv']);
writetable(T, f);
R.files.tables{end+1} = f;
R.files.tables{end+1} = fullfile(texDir, [base '.tex']);
end

function write_tex_table(fname, caption, label, colspec, header, rows, fontcmd)
fid = fopen(fname, 'w');
if fid < 0, error('Cannot open %s for writing.', fname); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%% Generated by generate_paper_results.m. Requires \\newcommand{\\trow}.\n');
fprintf(fid, '\\begin{table}[ht]\n\\caption{%s}\n\\label{%s}\n\\begin{center}\n%s\n\\begin{tabular}{%s}\n\\hline\n', ...
    caption, label, fontcmd, colspec);
fprintf(fid, '\\trow %s \\\\\n\\hline\\hline\n', strjoin(header, ' & '));
for r = 1:size(rows, 1)
    fprintf(fid, '\\trow %s \\\\\n\\hline\n', strjoin(rows(r,:), ' & '));
end
fprintf(fid, '\\end{tabular}\n\\end{center}\n\\end{table}\n');
end

function s = num2tex(x, spec)
if nargin < 2, spec = '%.3g'; end
if isnan(x), s = '--'; return; end
if isinf(x), s = '$\infty$'; return; end
s = sprintf(spec, x);
tok = regexp(s, '^(-?[\d\.]+)e([+-]\d+)$', 'tokens', 'once');
if ~isempty(tok)
    s = sprintf('$%s\\times10^{%d}$', tok{1}, sscanf(tok{2}, '%d'));
elseif ~isempty(s) && s(1) == '-'
    s = ['$-$' s(2:end)];
end
end

function s = yesno(tf)
if tf, s = 'Yes'; else, s = 'No'; end
end

function s = sanitize(str)
s = regexprep(str, '[^A-Za-z0-9]+', '_');
s = regexprep(s, '^_+|_+$', '');
end

%% =========================================================================
%                               FIGURES
%% =========================================================================
function [files, F] = fig_formation(out, c, o, figDir)
nC = numel(out.dr_rtn);
T0 = 2*pi*sqrt(out.states.chief(1,1)^3 / o.mu_km3s2);
iOrb = out.t(:) <= T0;
fig = new_fig(o.DoubleColumnIn, 3.3, o);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) uncontrolled RTN trajectories over one orbit
ax1 = nexttile(tl); hold(ax1, 'on');
for j = 1:nC
    s = sty(j);
    r = 1000*out.dr_rtn{j}(iOrb,:);
    plot3(ax1, r(:,2), r(:,1), r(:,3), 'Color', s.color, 'LineStyle', s.ls, ...
        'LineWidth', 1.25, 'DisplayName', sprintf('Collector %d', j));
    plot3(ax1, r(1,2), r(1,1), r(1,3), s.mk, 'MarkerSize', 5, 'MarkerFaceColor', s.color, ...
        'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
plot3(ax1, 0, 0, 0, 'kp', 'MarkerFaceColor', 'k', 'MarkerSize', 7, 'DisplayName', 'Combiner');
xlabel(ax1, 'Tangential [m]'); ylabel(ax1, 'Radial [m]'); zlabel(ax1, 'Normal [m]');
axis(ax1, 'equal'); view(ax1, 35, 25); style_ax(ax1, o); panel_label(ax1, '(a)', o);
make_legend(ax1, o, 'northeast');

% (b) design-epoch geometry and direction to the star
ax2 = nexttile(tl); hold(ax2, 'on');
P = zeros(nC, 3);
for j = 1:nC, P(j,:) = 1000*out.dr_rtn{j}(1,:); end
uc0 = out.states.chief(1,6);
sv = [cos(c.m.beta)*cos(c.m.phi0 - uc0); cos(c.m.beta)*sin(c.m.phi0 - uc0); sin(c.m.beta)];
rho = mean(sqrt(sum(P.^2, 2)));
L = 1.5*rho;
edges = P([1:nC 1], :);
plot3(ax2, edges(:,2), edges(:,1), edges(:,3), 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
for j = 1:nC
    s = sty(j);
    plot3(ax2, P(j,2), P(j,1), P(j,3), s.mk, 'MarkerSize', 6, 'MarkerFaceColor', s.color, ...
        'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Collector %d', j));
end
plot3(ax2, 0, 0, 0, 'kp', 'MarkerFaceColor', 'k', 'MarkerSize', 7, 'DisplayName', 'Combiner');
quiver3(ax2, 0, 0, 0, L*sv(2), L*sv(1), L*sv(3), 0, 'Color', 'k', 'LineWidth', 1.5, ...
    'MaxHeadSize', 0.5, 'DisplayName', 'Direction to star');
xlabel(ax2, 'Tangential [m]'); ylabel(ax2, 'Radial [m]'); zlabel(ax2, 'Normal [m]');
axis(ax2, 'equal'); view(ax2, 35, 25); style_ax(ax2, o); panel_label(ax2, '(b)', o);
make_legend(ax2, o, 'northeast');

F = struct();
F.designRanges_m = sqrt(sum(P.^2, 2)).';
if nC >= 3
    nrm = cross(P(2,:) - P(1,:), P(3,:) - P(1,:));
    nrm = nrm/norm(nrm);
    F.planeNormalToStar_deg = acosd(min(1, abs(dot(nrm, sv.'))));
    F.baselines_m = [norm(P(1,:)-P(2,:)), norm(P(1,:)-P(3,:)), norm(P(2,:)-P(3,:))];
end
files = export_fig_local(fig, fullfile(figDir, 'fig_formation'), o.Resolution, true, o);
end

function files = fig_uncontrolled(cases, o, figDir)
fig = new_fig(o.DoubleColumnIn, 3.0, o);
ax = axes(fig); hold(ax, 'on');
seen = zeros(0,2); allv = [];
for i = 1:numel(cases)
    m = cases{i}.m; d = cases{i}.d;
    key = [m.ra, m.dec];
    if any(all(abs(seen - key) < 1e-9, 2)), continue; end
    seen(end+1,:) = key; %#ok<AGROW>
    s = sty(size(seen,1));
    th = d.outT_abs_s/3600;
    plot(ax, th, d.outSpread_m, 'Color', s.color, 'LineStyle', s.ls, 'LineWidth', 1.25, ...
        'DisplayName', m.target);
    allv = [allv; d.outSpread_m(:)]; %#ok<AGROW>
end
set(ax, 'YScale', 'log');
style_ax(ax, o);
allv = allv(isfinite(allv) & allv > 0);
xlim(ax, [0 max(cellfun(@(x) x.d.outT_abs_s(end), cases))/3600]);
ylim(ax, [10^floor(log10(min(allv))), 10^ceil(log10(max(allv)*1.5))]);
seen = zeros(0,2);
for i = 1:numel(cases)
    m = cases{i}.m;
    key = [m.ra, m.dec];
    if any(all(abs(seen - key) < 1e-9, 2)), continue; end
    seen(end+1,:) = key; %#ok<AGROW>
    plot(ax, m.t0_s/3600, m.uncSpread0_m, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5, 'HandleVisibility', 'off');
    tsci = (m.t0_s + cases{i}.d.t_s(cases{i}.d.k_s))/3600;
    plot(ax, tsci, m.uncSpreadSci0_m, 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 5, 'HandleVisibility', 'off');
end
plot(ax, NaN, NaN, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5, 'DisplayName', 'Handoff');
plot(ax, NaN, NaN, 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 5, 'DisplayName', 'Science start');
xl = xlim(ax);
plot(ax, xl, [o.Dmax_m o.Dmax_m], 'k-.', 'LineWidth', 1, 'DisplayName', 'D_{max}');
xlabel(ax, 'Time since epoch [h]'); ylabel(ax, 'Uncontrolled OPD spread [m]');
make_legend(ax, o, 'best');
files = export_fig_local(fig, fullfile(figDir, 'fig_uncontrolled_opd'), o.Resolution, true, o);
end

function files = fig_diffpert(rep, o, figDir)
names = {'J2','SRP','Moon','Sun'};
pk = rep.peaks; Nc = max(pk.Collector);
M = nan(4, Nc); src = cellstr(pk.Source);
for s = 1:4
    for j = 1:Nc
        idx = strcmp(src, names{s}) & (pk.Collector == j);
        if any(idx), M(s,j) = pk.PeakDifferential_nmps2(find(idx,1)); end
    end
end
fig = new_fig(o.SingleColumnIn, 3.0, o);
ax = axes(fig); hold(ax, 'on');
pos = M(isfinite(M) & M > 0);
lo = 10^floor(log10(min(pos)));
hi = 10^ceil(log10(max(pos)*1.2));
if hi <= lo, hi = lo*10; end
P = M; P(P <= 0) = NaN;
fc = [0.00 0.27 0.53; 0.84 0.37 0.00; 0.80 0.90 0.95];
hb = bar(ax, P, 'grouped', 'BarWidth', 0.85);
for j = 1:numel(hb)
    hb(j).FaceColor = fc(mod(j-1,3)+1, :);
    hb(j).EdgeColor = 'k';
    hb(j).LineWidth = 0.75;
    hb(j).BaseValue = lo;
    hb(j).DisplayName = sprintf('Collector %d', j);
end
set(ax, 'YScale', 'log', 'YLim', [lo hi], 'XLim', [0.5 4.5], 'XTick', 1:4, ...
    'XTickLabel', {'Earth J_2', 'SRP', 'Moon', 'Sun'});
style_ax(ax, o);
ylabel(ax, 'Peak differential acceleration [nm s^{-2}]');
make_legend(ax, o, 'northwest');
files = export_fig_local(fig, fullfile(figDir, 'fig_diffpert'), o.Resolution, true, o);
end

function files = fig_ocp_opd(c, o, figDir, tag)
d = c.d; t_h = d.t_s/3600; Nc = size(d.relOPD_m, 2); xs = t_h(d.k_s);
fig = new_fig(o.DoubleColumnIn, 5.4, o);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) controlled vs uncontrolled spread
ax = nexttile(tl); hold(ax, 'on');
plot(ax, t_h, d.unc_m, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.25, 'DisplayName', 'Uncontrolled');
plot(ax, t_h, d.spread_m, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Controlled');
set(ax, 'YScale', 'log'); style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]);
v = [d.unc_m; d.spread_m]; v = v(isfinite(v) & v > 0);
ylim(ax, [10^floor(log10(min(v))), 10^ceil(log10(max(v)*1.5))]);
xl = xlim(ax);
plot(ax, xl, [o.Dmax_m o.Dmax_m], 'k-.', 'LineWidth', 1, 'DisplayName', 'D_{max}');
mark_science(ax, xs, true);
ylabel(ax, 'OPD spread [m]'); panel_label(ax, '(a)', o); make_legend(ax, o, 'best');

% (b) per-collector OPD relative to the mean
ax = nexttile(tl); hold(ax, 'on');
for j = 1:Nc
    s = sty(j);
    plot_marked(ax, t_h, d.relOPD_m(:,j), s, sprintf('Collector %d', j));
end
style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]);
mark_science(ax, xs, false);
ylabel(ax, 'OPD relative to mean [m]'); panel_label(ax, '(b)', o); make_legend(ax, o, 'best');

% (c) external and relay contributions to the spread
ax = nexttile(tl); hold(ax, 'on');
plot(ax, t_h, d.extSpread_m,   'k-',  'LineWidth', 1.5, 'DisplayName', 'Projection term');
plot(ax, t_h, d.relaySpread_m, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'Relay term');
set(ax, 'YScale', 'log'); style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]);
mark_science(ax, xs, false);
xlabel(ax, 'Time since handoff [h]'); ylabel(ax, 'Spread of each term [m]');
panel_label(ax, '(c)', o); make_legend(ax, o, 'best');

% (d) collector-to-combiner distance
ax = nexttile(tl); hold(ax, 'on');
for j = 1:Nc
    s = sty(j);
    plot_marked(ax, t_h, d.range_km(:,j), s, sprintf('Collector %d', j));
end
style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]);
mark_science(ax, xs, false);
xlabel(ax, 'Time since handoff [h]'); ylabel(ax, 'Collector-to-combiner distance [km]');
panel_label(ax, '(d)', o); make_legend(ax, o, 'best');
files = export_fig_local(fig, fullfile(figDir, ['fig_ocp_opd_' tag]), o.Resolution, true, o);
end

function files = fig_ocp_controls(c, o, figDir, tag)
d = c.d; t_h = d.t_s/3600; tc = t_h(1:end-1); Nc = size(d.alpha, 2); xs = t_h(d.k_s);
fig = new_fig(o.DoubleColumnIn, 6.2, o);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
lab = {'(a)','(b)','(c)','(d)','(e)','(f)'};

ax = nexttile(tl); hold(ax, 'on');
for j = 1:Nc
    s = sty(j);
    stairs(ax, tc, d.alpha(:,j), 'Color', s.color, 'LineStyle', s.ls, 'LineWidth', 1.25, 'DisplayName', sprintf('Collector %d', j));
end
style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]); mark_science(ax, xs, false);
ylabel(ax, 'Throttle \alpha'); panel_label(ax, lab{1}, o); make_legend(ax, o, 'best');

ax = nexttile(tl); hold(ax, 'on');
for j = 1:Nc
    s = sty(j);
    plot(ax, t_h, d.massUsedT(:,j), 'Color', s.color, 'LineStyle', s.ls, 'LineWidth', 1.25);
end
style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]); mark_science(ax, xs, false);
ylabel(ax, 'Propellant used [kg]'); panel_label(ax, lab{2}, o);

comp = {'d_R','d_T','d_N'};
for q = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    for j = 1:Nc
        s = sty(j);
        stairs(ax, tc, squeeze(d.dir(:,q,j)), 'Color', s.color, 'LineStyle', s.ls, 'LineWidth', 1.25);
    end
    style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]); ylim(ax, [-1.05 1.05]);
    mark_science(ax, xs, false);
    ylabel(ax, ['Thrust direction ' comp{q}]); panel_label(ax, lab{2+q}, o);
    if q >= 2, xlabel(ax, 'Time since handoff [h]'); end
end

ax = nexttile(tl); hold(ax, 'on');
for j = 1:Nc
    s = sty(j);
    plot(ax, t_h, d.dvCum(:,j), 'Color', s.color, 'LineStyle', s.ls, 'LineWidth', 1.25);
end
style_ax(ax, o); xlim(ax, [t_h(1) t_h(end)]); mark_science(ax, xs, false);
xlabel(ax, 'Time since handoff [h]'); ylabel(ax, 'Cumulative \Deltav [m s^{-1}]');
panel_label(ax, lab{6}, o);
files = export_fig_local(fig, fullfile(figDir, ['fig_ocp_controls_' tag]), o.Resolution, true, o);
end

function files = fig_uv(c, o, figDir, tag)
fig = new_fig(o.SingleColumnIn, 3.7, o);
ax = axes(fig); hold(ax, 'on');
plot_uv_tracks(ax, c.uv, true);
axis(ax, 'equal'); style_ax(ax, o);
xlabel(ax, ['u [' c.uv.unit ']']); ylabel(ax, ['v [' c.uv.unit ']']);
lg = legend(ax, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'off');
lg.FontSize = o.FontSize - 1; lg.FontName = o.FontName;
files = export_fig_local(fig, fullfile(figDir, ['fig_uv_' tag]), o.Resolution, true, o);
end

function files = fig_uv_all(cases, o, figDir)
n = numel(cases); nc = min(n, 3); nr = ceil(n/nc);
fig = new_fig(o.DoubleColumnIn, 2.5*nr + 0.3, o);
tl = tiledlayout(fig, nr, nc, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:n
    ax = nexttile(tl); hold(ax, 'on');
    plot_uv_tracks(ax, cases{i}.uv, false);
    axis(ax, 'equal'); style_ax(ax, o);
    xlabel(ax, ['u [' cases{i}.uv.unit ']']); ylabel(ax, ['v [' cases{i}.uv.unit ']']);
    t = title(ax, sprintf('(%c) %s', 'a'+i-1, cases{i}.m.label), 'FontName', o.FontName, ...
        'FontSize', o.FontSize, 'FontWeight', 'normal');
    try, ax.TitleHorizontalAlignment = 'left'; catch, set(t, 'HorizontalAlignment', 'left'); end %#ok<NOCOM>
end
files = export_fig_local(fig, fullfile(figDir, 'fig_uv_all'), o.Resolution, true, o);
end

function plot_uv_tracks(ax, uv, withLegend)
Np = size(uv.uv_m, 3);
for q = 1:Np
    s = sty(q);
    x = uv.scale*uv.uv_m(:,1,q); y = uv.scale*uv.uv_m(:,2,q);
    h1 = plot(ax, x, y, 'Color', s.color, 'LineStyle', '-', 'LineWidth', 1.5);
    if withLegend
        h1.DisplayName = sprintf('Baseline %d-%d', uv.pairs(q,1), uv.pairs(q,2));
    else
        h1.HandleVisibility = 'off';
    end
    plot(ax, -x, -y, 'Color', s.color, 'LineStyle', '--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    plot(ax, x(1), y(1), s.mk, 'MarkerSize', 5, 'MarkerFaceColor', s.color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    plot(ax, -x(1), -y(1), s.mk, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', s.color, 'HandleVisibility', 'off');
end
end

function files = fig_sky(S, cases, o, figDir)
fig = new_fig(o.DoubleColumnIn, 3.9, o);
ax = axes(fig); hold(ax, 'on');
mk = max(2, round(1800/sqrt(S.nTargets)));
scatter(ax, S.RA, S.DEC, mk, S.cov, 'filled', 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
colormap(ax, parula(256));
try, clim(ax, [0 100]); catch, caxis(ax, [0 100]); end %#ok<NOCOM,CAXIS>
cb = colorbar(ax, 'eastoutside');
cb.Label.String = 'Accessible time [% of year]';
cb.FontName = o.FontName; cb.FontSize = o.FontSize;

eps_obl = 23.4393;
lam = linspace(0, 2*pi, 3601);
xe = cos(lam); ye = sin(lam)*cosd(eps_obl); ze = sin(lam)*sind(eps_obl);
[ra_e, ord] = sort(mod(atan2d(ye, xe), 360));
plot(ax, ra_e, asind(ze(ord)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ecliptic');
A_G = [-0.0548755604, -0.8734370902, -0.4838350155; ...
        0.4941094279, -0.4448296300,  0.7469822445; ...
       -0.8676661490, -0.1980763734,  0.4559837762];
bl = linspace(0, 2*pi, 3601);
vg = A_G.' * [cos(bl); sin(bl); zeros(size(bl))];
[ra_g, ord] = sort(mod(atan2d(vg(2,:), vg(1,:)), 360));
plot(ax, ra_g, asind(vg(3,ord)), 'w-', 'LineWidth', 2.5, 'HandleVisibility', 'off');
plot(ax, ra_g, asind(vg(3,ord)), 'k-.', 'LineWidth', 1.25, 'DisplayName', 'Galactic plane');
plot(ax, 266.40, -28.94, 'kp', 'MarkerFaceColor', 'w', 'MarkerSize', 9, 'DisplayName', 'Galactic center');
seen = zeros(0,2);
for i = 1:numel(cases)
    m = cases{i}.m; key = [m.ra, m.dec];
    if any(all(abs(seen - key) < 1e-9, 2)), continue; end
    seen(end+1,:) = key; %#ok<AGROW>
    s = sty(size(seen,1));
    plot(ax, mod(rad2deg(m.ra), 360), rad2deg(m.dec), s.mk, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', m.target);
end
set(ax, 'XLim', [0 360], 'YLim', [-90 90], 'XTick', 0:60:360, 'YTick', -90:30:90);
style_ax(ax, o);
xlabel(ax, 'Right ascension [deg]'); ylabel(ax, 'Declination [deg]');
lg = legend(ax, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'off');
lg.FontSize = o.FontSize - 1; lg.FontName = o.FontName;
files = export_fig_local(fig, fullfile(figDir, 'fig_sky'), 300, false, o);
end

%% --------------------------------------------------------------- plot helpers
function fig = new_fig(w, h, o)
fig = figure('Units', 'inches', 'Position', [1 1 w h], 'Color', 'w', 'Visible', o.Visible);
set(fig, 'PaperUnits', 'inches', 'PaperSize', [w h], 'PaperPosition', [0 0 w h]);
end

function style_ax(ax, o)
set(ax, 'FontName', o.FontName, 'FontSize', o.FontSize, 'LineWidth', 0.75, 'Box', 'on', ...
    'TickDir', 'in', 'Color', 'w', 'XGrid', 'off', 'YGrid', 'off', 'Layer', 'top');
end

function panel_label(ax, str, o)
t = title(ax, str, 'FontName', o.FontName, 'FontSize', o.FontSize + 1, 'FontWeight', 'normal');
try, ax.TitleHorizontalAlignment = 'left'; catch, set(t, 'HorizontalAlignment', 'left'); end %#ok<NOCOM>
end

function make_legend(ax, o, loc)
lg = legend(ax, 'Location', loc, 'Box', 'off');
lg.FontName = o.FontName; lg.FontSize = o.FontSize - 1;
end

function s = sty(j)
C = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.4660 0.6740 0.1880; ...
     0.4940 0.1840 0.5560; 0.9290 0.6940 0.1250];
LS = {'-', '--', ':', '-.', '-'};
MK = {'o', 's', '^', 'd', 'v'};
k = mod(j-1, 5) + 1;
s.color = C(k,:); s.ls = LS{k}; s.mk = MK{k};
end

function plot_marked(ax, x, y, s, name)
mi = unique(round(linspace(1, numel(x), 8)));
plot(ax, x, y, 'Color', s.color, 'LineStyle', s.ls, 'LineWidth', 1.25, 'Marker', s.mk, ...
    'MarkerSize', 4, 'MarkerIndices', mi, 'MarkerFaceColor', 'w', 'DisplayName', name);
end

function mark_science(ax, x, showLegend)
xl = xlim(ax); yl = ylim(ax); xlim(ax, xl); ylim(ax, yl);
h = plot(ax, [x x], yl, 'k:', 'LineWidth', 1.0);
if showLegend, set(h, 'DisplayName', 'Science start'); else, set(h, 'HandleVisibility', 'off'); end
end

function files = export_fig_local(fig, base, res, vector, o) %#ok<INUSD>
pdf = [base '.pdf']; png = [base '.png'];
if vector
    exportgraphics(fig, pdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
else
    exportgraphics(fig, pdf, 'ContentType', 'image', 'Resolution', res, 'BackgroundColor', 'white');
end
exportgraphics(fig, png, 'Resolution', res, 'BackgroundColor', 'white');
files = {pdf, png};
for k = 1:2
    info = dir(files{k});
    if ~isempty(info) && info.bytes > 3e6
        warning('generate_paper_results:size', '%s is %.1f MB (SPIE asks for 2-3 MB per figure).', ...
            files{k}, info.bytes/1e6);
    end
end
close(fig);
end