function metrics = formation_opd_metrics(out)
% formation_opd_metrics
%
% Compute formation-level OPD metrics from the pipeline "out" structure.
%
% Supports:
%   - single deputy output
%   - multi-deputy output (cell array form)
%
% Inputs:
%   out : struct returned by opd_pipeline_from_qns
%
% Outputs:
%   metrics : struct containing time histories and scalar summaries
%
% Main outputs:
%   metrics.N                  number of collectors/deputies
%   metrics.opdMatrix          [Nt x N] OPD matrix [km]
%   metrics.meanOPD            [Nt x 1] mean OPD [km]
%   metrics.rangeOPD           [Nt x 1] max-min OPD [km]
%   metrics.maxPairwiseOPD     [Nt x 1] max pairwise OPD difference [km]
%   metrics.rmsRelativeOPD     [Nt x 1] RMS OPD relative to mean [km]
%   metrics.opdRelative        1xN cell, each [Nt x 1], relative to mean [km]
%
%   metrics.summary.maxRangeOPD_km
%   metrics.summary.maxRangeOPD_m
%   metrics.summary.maxPairwiseOPD_km
%   metrics.summary.maxPairwiseOPD_m
%   metrics.summary.maxRMSRelativeOPD_km
%   metrics.summary.maxRMSRelativeOPD_m
%   metrics.summary.maxAbsRelativePerCollector_km
%   metrics.summary.maxAbsRelativePerCollector_m
%
% Notes:
%   - For a single deputy, formation metrics are still returned, but
%     pairwise/range metrics are zero because there is only one beam.
%   - Absolute OPD is less meaningful than relative OPD for beam combining;
%     use range/maxPairwise/RMS-relative metrics for hardware sizing.
%

% -------------------------------------------------------------------------
% Convert out.opd to matrix form [Nt x N]
% -------------------------------------------------------------------------
if iscell(out.opd)
    N = numel(out.opd);
    Nt = length(out.opd{1});
    opdMatrix = zeros(Nt, N);
    for k = 1:N
        opdMatrix(:,k) = out.opd{k}(:);
    end
else
    N = 1;
    opdMatrix = out.opd(:);
    Nt = length(opdMatrix);
end

% -------------------------------------------------------------------------
% Mean OPD
% -------------------------------------------------------------------------
meanOPD = mean(opdMatrix, 2);

% -------------------------------------------------------------------------
% Relative OPD per collector
% -------------------------------------------------------------------------
opdRelative = cell(1, N);
for k = 1:N
    opdRelative{k} = opdMatrix(:,k) - meanOPD;
end

% -------------------------------------------------------------------------
% Range and pairwise metrics
% -------------------------------------------------------------------------
if N == 1
    rangeOPD = zeros(Nt,1);
    maxPairwiseOPD = zeros(Nt,1);
    rmsRelativeOPD = zeros(Nt,1);
else
    maxOPD = max(opdMatrix, [], 2);
    minOPD = min(opdMatrix, [], 2);
    rangeOPD = maxOPD - minOPD;

    % For scalar OPD values at each time, max pairwise difference = range
    maxPairwiseOPD = rangeOPD;

    % RMS relative to mean
    rmsRelativeOPD = sqrt(mean((opdMatrix - meanOPD).^2, 2));
end

% -------------------------------------------------------------------------
% Collector-level summary
% -------------------------------------------------------------------------
maxAbsRelativePerCollector = zeros(1,N);
for k = 1:N
    maxAbsRelativePerCollector(k) = max(abs(opdRelative{k}));
end

% -------------------------------------------------------------------------
% Package outputs
% -------------------------------------------------------------------------
metrics = struct();
metrics.N = N;
metrics.opdMatrix = opdMatrix;             % [km]
metrics.meanOPD = meanOPD;                 % [km]
metrics.rangeOPD = rangeOPD;               % [km]
metrics.maxPairwiseOPD = maxPairwiseOPD;   % [km]
metrics.rmsRelativeOPD = rmsRelativeOPD;   % [km]
metrics.opdRelative = opdRelative;         % cell, [km]

metrics.summary = struct();
metrics.summary.maxRangeOPD_km = max(rangeOPD);
metrics.summary.maxRangeOPD_m  = 1000 * max(rangeOPD);

metrics.summary.maxPairwiseOPD_km = max(maxPairwiseOPD);
metrics.summary.maxPairwiseOPD_m  = 1000 * max(maxPairwiseOPD);

metrics.summary.maxRMSRelativeOPD_km = max(rmsRelativeOPD);
metrics.summary.maxRMSRelativeOPD_m  = 1000 * max(rmsRelativeOPD);

metrics.summary.maxAbsRelativePerCollector_km = maxAbsRelativePerCollector;
metrics.summary.maxAbsRelativePerCollector_m  = 1000 * maxAbsRelativePerCollector;

[~, idxWorst] = max(maxAbsRelativePerCollector);
metrics.summary.worstCollectorIndex = idxWorst;

end