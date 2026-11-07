function [kStart, info] = find_science_hold_start( ...
    out, Dmax_m, orbitPeriod_s, settle_s, Tint_s)
% Select a propulsion handoff ahead of a later science interval.
% out.OPDfull_km is Nt-by-Nc: -dr.s + ||dr||, in km.
% out.t is Nt-by-1 in seconds. All time arguments are seconds.
%
% kStart is a SAMPLE INDEX at or after one orbit; its state initializes
% the OCP. info.scienceStartTime_s is the intended start of science and
% info.settleDuration_s is the actual interval between these two times.
% This coast-based selection is a guess: only a successful controlled
% trajectory can establish feasibility through the full science interval.

t = out.t(:);
path = out.OPDfull_km;
if size(path,1) ~= numel(t) || size(path,2) < 2
    error('out.OPDfull_km must be Nt-by-Nc with at least two collectors.');
end
if numel(t) < 3 || any(~isfinite(t)) || any(diff(t) <= 0) || ...
        any(~isfinite(path(:)))
    error('The path and strictly increasing time history must be finite.');
end
if ~isscalar(Dmax_m) || ~isscalar(orbitPeriod_s) || ...
        ~isscalar(settle_s) || ~isscalar(Tint_s) || ...
        any(~isfinite([Dmax_m orbitPeriod_s settle_s Tint_s])) || ...
        Dmax_m <= 0 || orbitPeriod_s <= 0 || ...
        settle_s <= 0 || Tint_s <= 0
    error('Dmax_m, orbitPeriod_s, settle_s and Tint_s must be positive scalars.');
end

spread_m = 1000 * (max(path,[],2) - min(path,[],2));
% Interior local minima represent alignment events, not merely the first
% sample below the limit. At least one orbit must elapse before handoff.
isMinimum = false(numel(t),1);
isMinimum(2:end-1) = ...
    spread_m(2:end-1) <= spread_m(1:end-2) & ...
    spread_m(2:end-1) <  spread_m(3:end);

alignments = find(isMinimum & spread_m <= 10*Dmax_m & ...
    t >= t(1) + settle_s + 3/4*orbitPeriod_s & ...
    t + Tint_s <= t(end));

kStart = [];
kScience = [];
for n = 1:numel(alignments)
    k = alignments(n);
    handoff = find(t <= t(k) - settle_s, 1, 'last');
    if ~isempty(handoff) && t(handoff) >= t(1)
        kStart = handoff;
        kScience = k;
        disp(kStart)
        break
    end
end
if isempty(kStart)
    error(['No qualifying realignment after one orbit leaves both ', ...
           'the requested settling interval and the full science interval.']);
end

info.scienceStartIndex = kScience;
info.scienceStartTime_s = t(kScience);
info.handoffTime_s = t(kStart);
info.settleDuration_s = t(kScience) - t(kStart);
info.scienceDuration_s = Tint_s;
info.handoffSpread_m = spread_m(kStart);
info.predictedAlignmentSpread_m = spread_m(kScience);
info.fullPathSpread_m = spread_m;
end
