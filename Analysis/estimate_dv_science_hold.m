function hold = estimate_dv_science_hold(t, out, opts)
% estimate_dv_science_hold
%
% Analytical estimate of delta-v needed to enter and maintain a science
% hold mode for an arbitrary number of collector satellites using the
% existing "out" pipeline structure.
%
% The science metric is the relative OPD spread across collectors:
%
%   OPD_range(t) = max_i(OPD_i) - min_i(OPD_i)
%
% A science hold is considered feasible when:
%
%   OPD_range(t) <= Dmax
%
% Inputs:
%   t    : Nx1 time vector [s]
%   out  : output struct from opd_pipeline_from_qns
%   opts : options struct with fields:
%
%       opts.Dmax_m             % OPD spread threshold [m], default 5
%       opts.integrationTimes_s % vector of desired hold durations [s]
%       opts.startMode          % 'bestWindow' | 'firstFeasible' |
%                               % 'specifiedIndex' | 'specifiedTime'
%       opts.startIndex         % index if startMode='specifiedIndex'
%       opts.startTime_s        % time if startMode='specifiedTime'
%       opts.entryTime_s        % assumed time to enter science hold [s], default 60
%       opts.method             % 'relativeOPD' (default)
%       opts.verbose            % true/false, default true
%
% Outputs:
%   hold : struct containing
%
%       hold.Ncollectors
%       hold.startIndex
%       hold.startTime_s
%       hold.startTime_orbits
%
%       hold.opdRange_m         % [Nx1] OPD spread [m]
%       hold.opdMean_m          % [Nx1] mean OPD [m]
%       hold.opdRelative_m      % 1xN cell of relative OPD histories [m]
%       hold.opdDotRelative_mps % 1xN cell of relative OPD rate histories [m/s]
%       hold.opdDDotRelative_mps2 % 1xN cell of relative OPD accel approx [m/s^2]
%
%       hold.passiveWindow_s
%       hold.passiveWindow_indices
%
%       hold.entryDV_perCollector_mps
%       hold.entryDV_max_mps
%       hold.entryDV_sum_mps
%
%       hold.integrationTimes_s
%       hold.maintDV_perCollector_mps   % [Ntint x N]
%       hold.maintDV_max_mps            % [Ntint x 1]
%       hold.maintDV_sum_mps            % [Ntint x 1]
%
%       hold.totalDV_max_mps            % entry + maintenance
%       hold.totalDV_sum_mps
%
% Notes:
%   - This is a first-order estimate, not an optimal control solution.
%   - Entry DV is approximated as a rest-to-rest 1D shift along the line of
%     sight needed to remove current relative OPD offsets in time T_entry:
%
%         DV_entry_i ~ 4 * |d_i| / T_entry
%
%   - Maintenance DV is approximated from the local relative OPD
%     second-derivative (finite-difference of OPD rate):
%
%         a_req_i ~ | d^2/dt^2 (OPD_i - mean(OPD)) |
%         DV_maint_i(T) ~ a_req_i * T
%
%     evaluated at hold start.
%

% -------------------------------------------------------------------------
% Defaults
% -------------------------------------------------------------------------
if nargin < 3
    opts = struct();
end

if ~isfield(opts,'Dmax_m'),             opts.Dmax_m = 5; end
if ~isfield(opts,'integrationTimes_s'), opts.integrationTimes_s = [300 600 900 1200 1800 3600]; end
if ~isfield(opts,'startMode'),          opts.startMode = 'bestWindow'; end
if ~isfield(opts,'entryTime_s'),        opts.entryTime_s = 60; end
if ~isfield(opts,'method'),             opts.method = 'relativeOPD'; end
if ~isfield(opts,'verbose'),            opts.verbose = true; end

if isvector(t), t = t(:); end
Nt = length(t);

% -------------------------------------------------------------------------
% Formation OPD metrics
% -------------------------------------------------------------------------
metrics = formation_opd_metrics(out);

N = metrics.N;
opdMatrix_m = 1000 * metrics.opdMatrix;     % convert km -> m
meanOPD_m   = 1000 * metrics.meanOPD;
rangeOPD_m  = 1000 * metrics.rangeOPD;

% Relative OPD per collector
opdRelative_m = cell(1,N);
for i = 1:N
    opdRelative_m{i} = 1000 * metrics.opdRelative{i};
end

% -------------------------------------------------------------------------
% Build OPD rate matrix from out.opd_dot
% -------------------------------------------------------------------------
opdDotMatrix_mps = zeros(Nt, N);
if iscell(out.opd_dot.total)
    for i = 1:N
        opdDotMatrix_mps(:,i) = 1000 * out.opd_dot.total{i}(:);
    end
else
    opdDotMatrix_mps(:,1) = 1000 * out.opd_dot.total(:);
end

meanOPDDot_mps = mean(opdDotMatrix_mps, 2);

opdDotRelative_mps = cell(1,N);
for i = 1:N
    opdDotRelative_mps{i} = opdDotMatrix_mps(:,i) - meanOPDDot_mps;
end

% -------------------------------------------------------------------------
% Approximate second derivative of relative OPD using finite difference
% -------------------------------------------------------------------------
opdDDotRelative_mps2 = cell(1,N);
for i = 1:N
    y = opdDotRelative_mps{i};
    ydd = zeros(size(y));

    if Nt >= 3
        % central difference interior
        ydd(2:Nt-1) = (y(3:Nt) - y(1:Nt-2)) ./ (t(3:Nt) - t(1:Nt-2));

        % one-sided ends
        ydd(1)  = (y(2) - y(1)) / (t(2) - t(1));
        ydd(end)= (y(end) - y(end-1)) / (t(end) - t(end-1));
    elseif Nt == 2
        ydd(:) = (y(2)-y(1)) / (t(2)-t(1));
    else
        ydd(:) = 0;
    end

    opdDDotRelative_mps2{i} = ydd;
end

% -------------------------------------------------------------------------
% Choose hold-start index
% -------------------------------------------------------------------------
switch lower(opts.startMode)
    case 'bestwindow'
        [~, k0] = min(rangeOPD_m);

    case 'firstfeasible'
        idx = find(rangeOPD_m <= opts.Dmax_m, 1, 'first');
        if isempty(idx)
            [~, idx] = min(rangeOPD_m);
        end
        k0 = idx;

    case 'specifiedindex'
        if ~isfield(opts,'startIndex')
            error('opts.startIndex required for startMode = specifiedIndex');
        end
        k0 = opts.startIndex;

    case 'specifiedtime'
        if ~isfield(opts,'startTime_s')
            error('opts.startTime_s required for startMode = specifiedTime');
        end
        [~, k0] = min(abs(t - opts.startTime_s));

    otherwise
        error('Unknown opts.startMode');
end

k0 = max(1, min(Nt, k0));
t0 = t(k0);

% -------------------------------------------------------------------------
% Passive science window duration from chosen start
% -------------------------------------------------------------------------
idxFail = find(rangeOPD_m(k0:end) > opts.Dmax_m, 1, 'first');

if isempty(idxFail)
    kEndPassive = Nt;
else
    kEndPassive = k0 + idxFail - 2;   % last still feasible
    kEndPassive = max(kEndPassive, k0);
end

passiveWindow_s = t(kEndPassive) - t(k0);

% -------------------------------------------------------------------------
% Entry DV estimate
% Bring all collectors to equal OPD by removing relative-to-mean OPD
% in an assumed entry time using a rest-to-rest 1D shift approximation:
%
%   DV ~ 4 * |d| / T_entry
%
% where d is the current relative OPD [m]
% -------------------------------------------------------------------------
Tentry = opts.entryTime_s;

entryDV_perCollector_mps = zeros(1,N);
opdRelative0_m = zeros(N,1);
opdDotRelative0_mps = zeros(N,1);
opdDDotRelative0_mps2 = zeros(N,1);

for i = 1:N
    opdRelative0_m(i)      = opdRelative_m{i}(k0);
    opdDotRelative0_mps(i) = opdDotRelative_mps{i}(k0);
    opdDDotRelative0_mps2(i)= opdDDotRelative_mps2{i}(k0);

    entryDV_perCollector_mps(i) = 4 * abs(opdRelative0_m(i)) / Tentry;
end

entryDV_max_mps = max(entryDV_perCollector_mps);
entryDV_sum_mps = sum(entryDV_perCollector_mps);

% -------------------------------------------------------------------------
% Maintenance DV estimate
%
% Approximate required control acceleration from local relative OPD
% second derivative at hold start:
%
%   a_req_i ~ |OPD_ddot_rel_i(t0)|
%
% Then:
%   DV_maint_i(T) ~ a_req_i * T
% -------------------------------------------------------------------------
Tint = opts.integrationTimes_s(:);
nTint = length(Tint);

maintDV_perCollector_mps = zeros(nTint, N);

for j = 1:nTint
    Tj = Tint(j);
    for i = 1:N
        a_req_i = abs(opdDDotRelative0_mps2(i));   % m/s^2
        maintDV_perCollector_mps(j,i) = a_req_i * Tj;
    end
end

maintDV_max_mps = max(maintDV_perCollector_mps, [], 2);
maintDV_sum_mps = sum(maintDV_perCollector_mps, 2);

% -------------------------------------------------------------------------
% Total DV estimate
% -------------------------------------------------------------------------
totalDV_max_mps = entryDV_max_mps + maintDV_max_mps;
totalDV_sum_mps = entryDV_sum_mps + maintDV_sum_mps;

% -------------------------------------------------------------------------
% Package outputs
% -------------------------------------------------------------------------
hold = struct();

hold.Ncollectors = N;

hold.startIndex = k0;
hold.startTime_s = t0;
if Nt >= 2
    % estimate orbit fraction from time span if user plotted in orbital time elsewhere
    hold.startTime_orbits = NaN;
else
    hold.startTime_orbits = NaN;
end

hold.opdRange_m = rangeOPD_m;
hold.opdMean_m  = meanOPD_m;
hold.opdRelative_m = opdRelative_m;
hold.opdDotRelative_mps = opdDotRelative_mps;
hold.opdDDotRelative_mps2 = opdDDotRelative_mps2;

hold.passiveWindow_s = passiveWindow_s;
hold.passiveWindow_indices = [k0, kEndPassive];

hold.opdRange0_m = rangeOPD_m(k0);
hold.opdMean0_m  = meanOPD_m(k0);
hold.opdRelative0_m = opdRelative0_m;
hold.opdDotRelative0_mps = opdDotRelative0_mps;
hold.opdDDotRelative0_mps2 = opdDDotRelative0_mps2;

hold.entryTime_s = Tentry;
hold.entryDV_perCollector_mps = entryDV_perCollector_mps;
hold.entryDV_max_mps = entryDV_max_mps;
hold.entryDV_sum_mps = entryDV_sum_mps;

hold.integrationTimes_s = Tint;
hold.maintDV_perCollector_mps = maintDV_perCollector_mps;
hold.maintDV_max_mps = maintDV_max_mps;
hold.maintDV_sum_mps = maintDV_sum_mps;

hold.totalDV_max_mps = totalDV_max_mps;
hold.totalDV_sum_mps = totalDV_sum_mps;

hold.method = opts.method;
hold.details = struct();
hold.details.note = ['Entry DV uses 1D rest-to-rest displacement approximation; ' ...
                     'maintenance DV uses local relative OPD second-derivative estimate.'];

% -------------------------------------------------------------------------
% Verbose summary
% -------------------------------------------------------------------------
if opts.verbose
    fprintf('=====================================================\n');
    fprintf(' Science Hold Delta-V Estimate\n');
    fprintf('=====================================================\n');
    fprintf('Collectors                : %d\n', N);
    fprintf('Start index               : %d\n', k0);
    fprintf('Start time                : %.3f s\n', t0);
    fprintf('Initial OPD range         : %.4f m\n', hold.opdRange0_m);
    fprintf('Passive window            : %.3f s\n', hold.passiveWindow_s);
    fprintf('Entry DV (max per sat)    : %.6e m/s\n', hold.entryDV_max_mps);
    fprintf('Entry DV (sum all sats)   : %.6e m/s\n', hold.entryDV_sum_mps);

    fprintf('\nIntegration time sweep:\n');
    fprintf('   T_int [s]     DV_maint_max [m/s]    DV_total_max [m/s]\n');
    for j = 1:nTint
        fprintf(' %10.2f      %14.6e      %14.6e\n', ...
            Tint(j), hold.maintDV_max_mps(j), hold.totalDV_max_mps(j));
    end
    fprintf('=====================================================\n\n');
end

end