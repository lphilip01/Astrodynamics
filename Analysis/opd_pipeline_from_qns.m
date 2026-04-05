function out = opd_pipeline_from_qns(t, xc, xd, paramsChief, paramsDeputy, mode, target)
% opd_pipeline_from_qns
%
% Complete QNS -> ROE -> RTN -> OPD / OPDdot pipeline with perturbation breakdown.
%
% Supports:
%   - single deputy: xd is Nx6, paramsDeputy is struct
%   - multiple deputies: xd is 1xNd cell, each cell Nx6
%                        paramsDeputy is 1xNd cell of structs
%
% For single deputy, output format is unchanged.
% For multiple deputies, output fields that depend on deputy are returned as cells:
%   out.roe{j}, out.roe_dot.J2{j}, out.dr_rtn{j}, out.opd{j}, out.opd_dot.J2{j}, ...
%

if isvector(t), t = t(:); end
N = length(t);

% Expand target if needed
if size(target,1) == 1
    target = repmat(target, N, 1);
elseif size(target,1) ~= N
    error('target must be 1x2 or Nx2 with same N as t.');
end

% Single-deputy mode
if ~iscell(xd)
    out = local_pipeline_single(t, xc, xd, paramsChief, paramsDeputy, mode, target);
    return
end

% Multi-deputy mode
nDep = numel(xd);

if ~iscell(paramsDeputy)
    error('If xd is a cell array, paramsDeputy must also be a cell array.');
end
if numel(paramsDeputy) ~= nDep
    error('xd and paramsDeputy cell arrays must have the same length.');
end

% Run single pipeline per deputy, then pack
singleOut = cell(1, nDep);
for j = 1:nDep
    singleOut{j} = local_pipeline_single(t, xc, xd{j}, paramsChief, paramsDeputy{j}, mode, target);
end

% Keep chief info once
out.chief = singleOut{1}.chief;

% Pack deputy-dependent fields into cells
out.roe = cell(1,nDep);
out.dr_rtn = cell(1,nDep);
out.opd = cell(1,nDep);
out.source_geom = struct('total',[],'J2',[],'SRP',[],'Moon',[],'Sun',[]);
out.roe_dot = struct('total',[],'J2',[],'SRP',[],'Moon',[],'Sun',[]);
out.opd_dot = struct('total',[],'J2',[],'SRP',[],'Moon',[],'Sun',[]);
out.deputy = struct('rates',struct('total',[],'J2',[],'SRP',[],'Moon',[],'Sun',[]));

out.roe_dot.total = cell(1,nDep);
out.roe_dot.J2    = cell(1,nDep);
out.roe_dot.SRP   = cell(1,nDep);
out.roe_dot.Moon  = cell(1,nDep);
out.roe_dot.Sun   = cell(1,nDep);

out.opd_dot.total = cell(1,nDep);
out.opd_dot.J2    = cell(1,nDep);
out.opd_dot.SRP   = cell(1,nDep);
out.opd_dot.Moon  = cell(1,nDep);
out.opd_dot.Sun   = cell(1,nDep);

out.deputy.rates.total = cell(1,nDep);
out.deputy.rates.J2    = cell(1,nDep);
out.deputy.rates.SRP   = cell(1,nDep);
out.deputy.rates.Moon  = cell(1,nDep);
out.deputy.rates.Sun   = cell(1,nDep);

out.source_geom.total = cell(1,nDep);
out.source_geom.J2    = cell(1,nDep);
out.source_geom.SRP   = cell(1,nDep);
out.source_geom.Moon  = cell(1,nDep);
out.source_geom.Sun   = cell(1,nDep);

for j = 1:nDep
    s = singleOut{j};
    out.roe{j} = s.roe;
    out.dr_rtn{j} = s.dr_rtn;
    out.opd{j} = s.opd;

    out.roe_dot.total{j} = s.roe_dot.total;
    out.roe_dot.J2{j}    = s.roe_dot.J2;
    out.roe_dot.SRP{j}   = s.roe_dot.SRP;
    out.roe_dot.Moon{j}  = s.roe_dot.Moon;
    out.roe_dot.Sun{j}   = s.roe_dot.Sun;

    out.opd_dot.total{j} = s.opd_dot.total;
    out.opd_dot.J2{j}    = s.opd_dot.J2;
    out.opd_dot.SRP{j}   = s.opd_dot.SRP;
    out.opd_dot.Moon{j}  = s.opd_dot.Moon;
    out.opd_dot.Sun{j}   = s.opd_dot.Sun;

    out.deputy.rates.total{j} = s.deputy.rates.total;
    out.deputy.rates.J2{j}    = s.deputy.rates.J2;
    out.deputy.rates.SRP{j}   = s.deputy.rates.SRP;
    out.deputy.rates.Moon{j}  = s.deputy.rates.Moon;
    out.deputy.rates.Sun{j}   = s.deputy.rates.Sun;

    out.source_geom.total{j} = s.source_geom.total;
    out.source_geom.J2{j}    = s.source_geom.J2;
    out.source_geom.SRP{j}   = s.source_geom.SRP;
    out.source_geom.Moon{j}  = s.source_geom.Moon;
    out.source_geom.Sun{j}   = s.source_geom.Sun;
end

end

% =========================================================================
function out = local_pipeline_single(t, xc, xd, paramsChief, paramsDeputy, mode, target)

N = length(t);

if size(xc,1) ~= N || size(xd,1) ~= N || size(xc,2) ~= 6 || size(xd,2) ~= 6
    error('xc and xd must be Nx6 with same N as t.');
end

chiefRates.total = zeros(N,6);
chiefRates.J2    = zeros(N,6);
chiefRates.SRP   = zeros(N,6);
chiefRates.Moon  = zeros(N,6);
chiefRates.Sun   = zeros(N,6);

deputyRates.total = zeros(N,6);
deputyRates.J2    = zeros(N,6);
deputyRates.SRP   = zeros(N,6);
deputyRates.Moon  = zeros(N,6);
deputyRates.Sun   = zeros(N,6);

roe       = zeros(N,6);
roeDotTot = zeros(N,6);
roeDotJ2  = zeros(N,6);
roeDotSRP = zeros(N,6);
roeDotMoon= zeros(N,6);
roeDotSun = zeros(N,6);

dr_rtn = zeros(N,3);
opd    = zeros(N,1);

opdDotTot  = zeros(N,1);
opdDotJ2   = zeros(N,1);
opdDotSRP  = zeros(N,1);
opdDotMoon = zeros(N,1);
opdDotSun  = zeros(N,1);

auxTotal = cell(N,1);
auxJ2    = cell(N,1);
auxSRP   = cell(N,1);
auxMoon  = cell(N,1);
auxSun   = cell(N,1);

for k = 1:N
    xc_k = xc(k,:).';
    xd_k = xd(k,:).';

    bc = qns_perturbation_breakdown(t(k), xc_k, paramsChief);
    bd = qns_perturbation_breakdown(t(k), xd_k, paramsDeputy);

    chiefRates.total(k,:) = bc.rates.total.';
    chiefRates.J2(k,:)    = bc.rates.J2.';
    chiefRates.SRP(k,:)   = bc.rates.SRP.';
    chiefRates.Moon(k,:)  = bc.rates.Moon.';
    chiefRates.Sun(k,:)   = bc.rates.Sun.';

    deputyRates.total(k,:) = bd.rates.total.';
    deputyRates.J2(k,:)    = bd.rates.J2.';
    deputyRates.SRP(k,:)   = bd.rates.SRP.';
    deputyRates.Moon(k,:)  = bd.rates.Moon.';
    deputyRates.Sun(k,:)   = bd.rates.Sun.';

    roe_k = roe_from_qns_chief_deputy(xc(k,:), xd(k,:));
    roe(k,:) = roe_k;

    roeDotTot(k,:)  = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.total.', bd.rates.total.');
    roeDotJ2(k,:)   = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.J2.',    bd.rates.J2.');
    roeDotSRP(k,:)  = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.SRP.',   bd.rates.SRP.');
    roeDotMoon(k,:) = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.Moon.',  bd.rates.Moon.');
    roeDotSun(k,:)  = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.Sun.',   bd.rates.Sun.');

    dr_rtn(k,:) = rtn_from_roe(roe_k, xc(k,1), xc(k,6));

    chiefGeomState.a    = xc(k,1);
    chiefGeomState.u    = xc(k,6);
    chiefGeomState.inc  = xc(k,4);
    chiefGeomState.RAAN = xc(k,5);

    [opd(k), ~] = opd_from_rtn(dr_rtn(k,:), mode, target(k,:), chiefGeomState);

    chiefGeomTot.a        = xc(k,1);
    chiefGeomTot.u        = xc(k,6);
    chiefGeomTot.a_dot    = bc.rates.total(1);
    chiefGeomTot.u_dot    = bc.rates.total(6);
    chiefGeomTot.inc      = xc(k,4);
    chiefGeomTot.RAAN     = xc(k,5);
    chiefGeomTot.inc_dot  = bc.rates.total(4);
    chiefGeomTot.RAAN_dot = bc.rates.total(5);
    [opdDotTot(k), auxTotal{k}] = opd_rate_from_roe(roe_k, roeDotTot(k,:), chiefGeomTot, mode, target(k,:));

    chiefGeomJ2 = chiefGeomTot;
    chiefGeomJ2.a_dot    = bc.rates.J2(1);
    chiefGeomJ2.u_dot    = bc.rates.J2(6);
    chiefGeomJ2.inc_dot  = bc.rates.J2(4);
    chiefGeomJ2.RAAN_dot = bc.rates.J2(5);
    [opdDotJ2(k), auxJ2{k}] = opd_rate_from_roe(roe_k, roeDotJ2(k,:), chiefGeomJ2, mode, target(k,:));

    chiefGeomSRP = chiefGeomTot;
    chiefGeomSRP.a_dot    = bc.rates.SRP(1);
    chiefGeomSRP.u_dot    = bc.rates.SRP(6);
    chiefGeomSRP.inc_dot  = bc.rates.SRP(4);
    chiefGeomSRP.RAAN_dot = bc.rates.SRP(5);
    [opdDotSRP(k), auxSRP{k}] = opd_rate_from_roe(roe_k, roeDotSRP(k,:), chiefGeomSRP, mode, target(k,:));

    chiefGeomMoon = chiefGeomTot;
    chiefGeomMoon.a_dot    = bc.rates.Moon(1);
    chiefGeomMoon.u_dot    = bc.rates.Moon(6);
    chiefGeomMoon.inc_dot  = bc.rates.Moon(4);
    chiefGeomMoon.RAAN_dot = bc.rates.Moon(5);
    [opdDotMoon(k), auxMoon{k}] = opd_rate_from_roe(roe_k, roeDotMoon(k,:), chiefGeomMoon, mode, target(k,:));

    chiefGeomSun = chiefGeomTot;
    chiefGeomSun.a_dot    = bc.rates.Sun(1);
    chiefGeomSun.u_dot    = bc.rates.Sun(6);
    chiefGeomSun.inc_dot  = bc.rates.Sun(4);
    chiefGeomSun.RAAN_dot = bc.rates.Sun(5);
    [opdDotSun(k), auxSun{k}] = opd_rate_from_roe(roe_k, roeDotSun(k,:), chiefGeomSun, mode, target(k,:));
end

out.roe = roe;

out.roe_dot.total = roeDotTot;
out.roe_dot.J2    = roeDotJ2;
out.roe_dot.SRP   = roeDotSRP;
out.roe_dot.Moon  = roeDotMoon;
out.roe_dot.Sun   = roeDotSun;

out.dr_rtn = dr_rtn;
out.opd    = opd;

out.opd_dot.total = opdDotTot;
out.opd_dot.J2    = opdDotJ2;
out.opd_dot.SRP   = opdDotSRP;
out.opd_dot.Moon  = opdDotMoon;
out.opd_dot.Sun   = opdDotSun;

out.chief.rates.total = chiefRates.total;
out.chief.rates.J2    = chiefRates.J2;
out.chief.rates.SRP   = chiefRates.SRP;
out.chief.rates.Moon  = chiefRates.Moon;
out.chief.rates.Sun   = chiefRates.Sun;

out.deputy.rates.total = deputyRates.total;
out.deputy.rates.J2    = deputyRates.J2;
out.deputy.rates.SRP   = deputyRates.SRP;
out.deputy.rates.Moon  = deputyRates.Moon;
out.deputy.rates.Sun   = deputyRates.Sun;

out.source_geom.total = auxTotal;
out.source_geom.J2    = auxJ2;
out.source_geom.SRP   = auxSRP;
out.source_geom.Moon  = auxMoon;
out.source_geom.Sun   = auxSun;

end