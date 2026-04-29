function [kStart, info] = find_science_hold_start(out, Dmax_m, mode)
% Find best start index for switching to science hold from out.opd
%
% Inputs:
%   out     - pipeline output struct with out.opd cell array
%   Dmax_m  - OPD spread threshold in meters
%   mode    - 'globalmin', 'firstfeasible', 'bestlocal'
%
% Outputs:
%   kStart  - selected start index
%   info    - struct with opdRange and candidates

if nargin < 3
    mode = 'bestlocal';
end

opdMat = cell2mat(out.opd(:)');   % [Nt x N]
opdRange = max(opdMat,[],2) - min(opdMat,[],2);   % km
dopdRange = gradient(opdRange);

Dmax_km = Dmax_m/1000;

switch lower(mode)
    case 'globalmin'
        [~, kStart] = min(opdRange);

    case 'firstfeasible'
        kStart = find(opdRange <= Dmax_km, 1, 'first');
        if isempty(kStart)
            [~, kStart] = min(opdRange);
        end

    case 'bestlocal'
        cand = find(islocalmin(opdRange));
        cand = cand(opdRange(cand) <= Dmax_km);

        if isempty(cand)
            [~, kStart] = min(opdRange);
        else
            score = opdRange(cand) + abs(dopdRange(cand));
            [~, ii] = min(score);
            kStart = cand(ii);
        end

    otherwise
        error('Unknown mode.');
end

info.opdRange_km = opdRange;
info.opdRange_m  = 1000*opdRange;
info.dopdRange_kmps = dopdRange;
info.threshold_m = Dmax_m;
end