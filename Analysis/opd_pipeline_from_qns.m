function out = opd_pipeline_from_qns(t, xc, xd, paramsChief, paramsDeputy, mode, target)
% opd_pipeline_from_qns
%
% Complete QNS -> ROE -> RTN -> OPD / OPDdot pipeline with perturbation breakdown.
%
% Inputs:
%   t            : Nx1 time vector [s]
%   xc           : Nx6 chief QNS state history
%   xd           : Nx6 deputy QNS state history
%
%   paramsChief  : params struct for chief (used by qns_perturbation_breakdown)
%   paramsDeputy : params struct for deputy
%
%   mode         : 'phibeta' or 'radec'
%
%   target       : target direction
%                  if mode='phibeta' : [phi0, beta] [rad], 1x2 or Nx2
%                  if mode='radec'   : [RA, Dec] [rad], 1x2 or Nx2
%
% Outputs:
%   out : struct containing
%
%     out.roe                    Nx6  ROE state
%     out.roe_dot.total          Nx6
%     out.roe_dot.J2             Nx6
%     out.roe_dot.SRP            Nx6
%     out.roe_dot.Moon           Nx6
%     out.roe_dot.Sun            Nx6
%
%     out.dr_rtn                 Nx3  RTN relative position [km]
%     out.opd                    Nx1  OPD [km]
%
%     out.opd_dot.total          Nx1  OPD rate [km/s]
%     out.opd_dot.J2             Nx1
%     out.opd_dot.SRP            Nx1
%     out.opd_dot.Moon           Nx1
%     out.opd_dot.Sun            Nx1
%
%     out.chief.rates.total      Nx6  chief QNS rates
%     out.chief.rates.J2         Nx6
%     out.chief.rates.SRP        Nx6
%     out.chief.rates.Moon       Nx6
%     out.chief.rates.Sun        Nx6
%
%     out.deputy.rates.total     Nx6  deputy QNS rates
%     out.deputy.rates.J2        Nx6
%     out.deputy.rates.SRP       Nx6
%     out.deputy.rates.Moon      Nx6
%     out.deputy.rates.Sun       Nx6
%
%     out.source_geom.total      auxiliary geometry struct from opd_rate_from_roe
%     out.source_geom.J2         same, for source-specific OPD rate
%     out.source_geom.SRP        ...
%     out.source_geom.Moon       ...
%     out.source_geom.Sun        ...
%
% Dependencies:
%   qns_perturbation_breakdown.m
%   roe_from_qns_chief_deputy.m
%   rtn_from_roe.m
%   opd_from_rtn.m
%   opd_rate_from_roe.m
%
% Notes:
%   - Uses the current propagated chief/deputy states as the reference states.
%   - Source-specific OPD rates are computed using source-specific chief geometry rates
%     and source-specific ROE rates.
%   - OPD itself is geometric and is not source-separated; only OPDdot is.
%

% -------------------------------------------------------------------------
% Input checks / shaping
% -------------------------------------------------------------------------
if isvector(t), t = t(:); end
N = length(t);

if size(xc,1) ~= N || size(xd,1) ~= N || size(xc,2) ~= 6 || size(xd,2) ~= 6
    error('xc and xd must be Nx6 with same N as t.');
end

if size(target,1) == 1
    target = repmat(target, N, 1);
elseif size(target,1) ~= N
    error('target must be 1x2 or Nx2 with same N as t.');
end

% -------------------------------------------------------------------------
% Preallocate
% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
% Main loop
% -------------------------------------------------------------------------
for k = 1:N
    xc_k = xc(k,:).';
    xd_k = xd(k,:).';

    % Chief / deputy perturbation breakdown at current state
    bc = qns_perturbation_breakdown(t(k), xc_k, paramsChief);
    bd = qns_perturbation_breakdown(t(k), xd_k, paramsDeputy);

    % Store QNS rates
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

    % ROE state from chief/deputy QNS states
    roe_k = roe_from_qns_chief_deputy(xc(k,:), xd(k,:));
    roe(k,:) = roe_k;

    % ROE rate from chief/deputy QNS rates, by perturbation
    roeDotTot(k,:)  = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.total.', bd.rates.total.');
    roeDotJ2(k,:)   = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.J2.',    bd.rates.J2.');
    roeDotSRP(k,:)  = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.SRP.',   bd.rates.SRP.');
    roeDotMoon(k,:) = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.Moon.',  bd.rates.Moon.');
    roeDotSun(k,:)  = qnsrates_to_roerates(xc(k,:), xd(k,:), bc.rates.Sun.',   bd.rates.Sun.');

    % RTN relative position
    dr_rtn(k,:) = rtn_from_roe(roe_k, xc(k,1), xc(k,6));

    % OPD (single geometric quantity from total current state)
    chiefGeomState.a    = xc(k,1);
    chiefGeomState.u    = xc(k,6);
    chiefGeomState.inc  = xc(k,4);
    chiefGeomState.RAAN = xc(k,5);

    [opd(k), ~] = opd_from_rtn(dr_rtn(k,:), mode, target(k,:), chiefGeomState);

    % Total OPD rate
    chiefGeomTot.a        = xc(k,1);
    chiefGeomTot.u        = xc(k,6);
    chiefGeomTot.a_dot    = bc.rates.total(1);
    chiefGeomTot.u_dot    = bc.rates.total(6);
    chiefGeomTot.inc      = xc(k,4);
    chiefGeomTot.RAAN     = xc(k,5);
    chiefGeomTot.inc_dot  = bc.rates.total(4);
    chiefGeomTot.RAAN_dot = bc.rates.total(5);

    [opdDotTot(k), auxTotal{k}] = opd_rate_from_roe(roe_k, roeDotTot(k,:), chiefGeomTot, mode, target(k,:));

    % J2 contribution to OPDdot
    chiefGeomJ2.a        = xc(k,1);
    chiefGeomJ2.u        = xc(k,6);
    chiefGeomJ2.a_dot    = bc.rates.J2(1);
    chiefGeomJ2.u_dot    = bc.rates.J2(6);
    chiefGeomJ2.inc      = xc(k,4);
    chiefGeomJ2.RAAN     = xc(k,5);
    chiefGeomJ2.inc_dot  = bc.rates.J2(4);
    chiefGeomJ2.RAAN_dot = bc.rates.J2(5);

    [opdDotJ2(k), auxJ2{k}] = opd_rate_from_roe(roe_k, roeDotJ2(k,:), chiefGeomJ2, mode, target(k,:));

    % SRP contribution
    chiefGeomSRP.a        = xc(k,1);
    chiefGeomSRP.u        = xc(k,6);
    chiefGeomSRP.a_dot    = bc.rates.SRP(1);
    chiefGeomSRP.u_dot    = bc.rates.SRP(6);
    chiefGeomSRP.inc      = xc(k,4);
    chiefGeomSRP.RAAN     = xc(k,5);
    chiefGeomSRP.inc_dot  = bc.rates.SRP(4);
    chiefGeomSRP.RAAN_dot = bc.rates.SRP(5);

    [opdDotSRP(k), auxSRP{k}] = opd_rate_from_roe(roe_k, roeDotSRP(k,:), chiefGeomSRP, mode, target(k,:));

    % Moon contribution
    chiefGeomMoon.a        = xc(k,1);
    chiefGeomMoon.u        = xc(k,6);
    chiefGeomMoon.a_dot    = bc.rates.Moon(1);
    chiefGeomMoon.u_dot    = bc.rates.Moon(6);
    chiefGeomMoon.inc      = xc(k,4);
    chiefGeomMoon.RAAN     = xc(k,5);
    chiefGeomMoon.inc_dot  = bc.rates.Moon(4);
    chiefGeomMoon.RAAN_dot = bc.rates.Moon(5);

    [opdDotMoon(k), auxMoon{k}] = opd_rate_from_roe(roe_k, roeDotMoon(k,:), chiefGeomMoon, mode, target(k,:));

    % Sun gravity contribution
    chiefGeomSun.a        = xc(k,1);
    chiefGeomSun.u        = xc(k,6);
    chiefGeomSun.a_dot    = bc.rates.Sun(1);
    chiefGeomSun.u_dot    = bc.rates.Sun(6);
    chiefGeomSun.inc      = xc(k,4);
    chiefGeomSun.RAAN     = xc(k,5);
    chiefGeomSun.inc_dot  = bc.rates.Sun(4);
    chiefGeomSun.RAAN_dot = bc.rates.Sun(5);

    [opdDotSun(k), auxSun{k}] = opd_rate_from_roe(roe_k, roeDotSun(k,:), chiefGeomSun, mode, target(k,:));
end

% -------------------------------------------------------------------------
% Package output
% -------------------------------------------------------------------------
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