function fig = animate_full_solution_presentation(out, sol, opts)
% animate_full_solution_presentation
%
% Presentation-focused synchronized animation that combines:
%   - collector formation in the star-normal plane view
%   - OPD relative to the formation mean
%   - collector RTN control histories
%
% The animation uses the same science-hold stitching logic as
% animate_full_solution.m, but renders the full solution in one aligned
% dashboard suitable for a science audience.
%
% Inputs:
%   out  : pipeline output struct from opd_pipeline_from_qns
%   sol  : solution struct from solve_science_hold_ocp
%   opts : optional struct
%
% Optional opts fields:
%   opts.Dmax_m         default 5
%   opts.holdStartMode  default 'bestLocal'
%   opts.holdStartIndex default []
%   opts.timeUnit       default 'hr'      ('s' | 'min' | 'hr')
%   opts.step           default 5
%   opts.pauseTime      default 0.03
%   opts.trailLength    default 30
%   opts.showPlane      default true
%   opts.axisEqual      default true
%   opts.figurePosition default [0.05 0.08 0.90 0.84]
%   opts.videoFile      default ''
%   opts.frameRate      default 20
%   opts.mode              default inferred from out.star
%   opts.target            default inferred from out.star
%   opts.dtSample_s        default median(diff(out.t))
%   opts.zoomLeadFrames    default 10  (# animation frames before OCP start
%                                       at which OPD/ctrl panels zoom in)
%   opts.ocpPauseMultiplier default 5  (pause multiplier during OCP window)
%
% Output:
%   fig : figure handle

if nargin < 3
    opts = struct();
end

if ~isfield(opts, 'Dmax_m'),         opts.Dmax_m = 5; end
if ~isfield(opts, 'holdStartMode'),  opts.holdStartMode = 'bestLocal'; end
if ~isfield(opts, 'holdStartIndex'), opts.holdStartIndex = []; end
if ~isfield(opts, 'timeUnit'),       opts.timeUnit = 'hr'; end
if ~isfield(opts, 'step'),           opts.step = 5; end
if ~isfield(opts, 'pauseTime'),      opts.pauseTime = 0.03; end
if ~isfield(opts, 'trailLength'),    opts.trailLength = 100; end
if ~isfield(opts, 'showPlane'),      opts.showPlane = true; end
if ~isfield(opts, 'axisEqual'),      opts.axisEqual = true; end
if ~isfield(opts, 'figurePosition'), opts.figurePosition = [0.05 0.08 0.90 0.84]; end
if ~isfield(opts, 'videoFile'),        opts.videoFile = ''; end
if ~isfield(opts, 'frameRate'),        opts.frameRate = 30; end
if ~isfield(opts, 'zoomLeadFrames'),   opts.zoomLeadFrames = 1; end
if ~isfield(opts, 'ocpPauseMultiplier'), opts.ocpPauseMultiplier = 1; end

opts.step = max(1, round(opts.step));
opts.trailLength = max(0, round(opts.trailLength));

if ~isfield(out, 't') || ~isfield(out, 'dr_rtn')
    error('out must contain fields t and dr_rtn.');
end
if ~iscell(out.dr_rtn)
    error('out.dr_rtn must be a cell array of collector RTN histories.');
end
if ~isfield(sol, 'RTN') || ~isfield(sol, 'U') || ~isfield(sol, 't')
    error('sol must contain RTN, U, and t.');
end

Nc = numel(out.dr_rtn);
if numel(sol.RTN) ~= Nc || numel(sol.U) ~= Nc
    error('The number of collectors in out and sol must match.');
end
if Nc < 3
    error('animate_full_solution_presentation requires at least 3 collectors.');
end

if isempty(opts.holdStartIndex)
    kStart = find_science_hold_start(out, opts.Dmax_m, opts.holdStartMode);
else
    kStart = opts.holdStartIndex;
end
kStart=kStart-3;

[mode, target] = local_resolve_target(out, opts);
data = local_build_full_history(out, sol, kStart, mode, target, opts);

colors = lines(Nc);
componentColors = [0.85 0.33 0.10;
                   0.00 0.45 0.74;
                   0.47 0.67 0.19];

frameList = 1:opts.step:data.Nt;
if frameList(end) ~= data.Nt
    frameList(end+1) = data.Nt;
end

% Find the index into frameList at which the animation cursor first reaches
% or passes the hold-start time, then step back zoomLeadFrames to get the
% frame at which we snap the OPD / ctrl panels into zoomed view.
holdStartIdx_inFull = find(data.tt >= data.holdStartTimePlot, 1, 'first');
if isempty(holdStartIdx_inFull)
    holdStartIdx_inFull = data.Nt;
end
holdStartFrame_inList = find(frameList >= holdStartIdx_inFull, 1, 'first');
if isempty(holdStartFrame_inList)
    holdStartFrame_inList = numel(frameList);
end
zoomTriggerFrame = max(1, holdStartFrame_inList - opts.zoomLeadFrames);

fig = figure( ...
    'Color', 'w', ...
    'Name', 'Full Solution Presentation Animation', ...
    'NumberTitle', 'off', ...
    'Units', 'normalized', ...
    'Position', opts.figurePosition);

axMain = axes('Parent', fig, 'Position', [0.06 0.33 0.66 0.60]);
axOpd  = axes('Parent', fig, 'Position', [0.06 0.08 0.91 0.17]);

ctrlLeft = 0.75;
ctrlWidth = 0.22;
ctrlBottom = 0.33;
ctrlHeightTotal = 0.60;
ctrlGap = 0.02;
ctrlHeight = (ctrlHeightTotal - ctrlGap*(Nc-1)) / Nc;
axCtrl = gobjects(1, Nc);

for j = 1:Nc
    yPos = ctrlBottom + (Nc-j) * (ctrlHeight + ctrlGap);
    axCtrl(j) = axes('Parent', fig, 'Position', [ctrlLeft yPos ctrlWidth ctrlHeight]);
end

% -------------------------------------------------------------------------
% Main formation view
% -------------------------------------------------------------------------
hold(axMain, 'on');
grid(axMain, 'on');
view(axMain, [-39.662250605979608,9.55629197688153]);
xlabel(axMain, 'R [km]');
ylabel(axMain, 'T [km]');
zlabel(axMain, 'N [km]');
title(axMain, 'Formation / Star Direction / Collector Plane');

if opts.axisEqual
    axis(axMain, 'equal');
end

xlim(axMain, [-data.mainLim data.mainLim]);
ylim(axMain, [-data.mainLim data.mainLim]);
zlim(axMain, [-data.mainLim data.mainLim]);

plot3(axMain, 0, 0, 0, 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 1);

hTrail = gobjects(1, Nc);
hCollector = gobjects(1, Nc);
for j = 1:Nc
    hTrail(j) = plot3(axMain, nan, nan, nan, '-', ...
        'Color', colors(j,:), 'LineWidth', 1.4);
    hCollector(j) = plot3(axMain, nan, nan, nan, 'o', ...
        'Color', colors(j,:), ...
        'MarkerFaceColor', colors(j,:), ...
        'MarkerSize', 8);
end

hConn = plot3(axMain, nan, nan, nan, 'k--', 'LineWidth', 1.0);
hPlane = patch(axMain, nan, nan, nan, [0.70 0.80 1.00], ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none', 'Visible', 'off');
hStar = quiver3(axMain, 0, 0, 0, 0, 0, 0, 0, ...
    'Color', [0.85 0.15 0.15], 'LineWidth', 2.2, 'MaxHeadSize', 0.5);
hStarText = text(axMain, 0, 0, 0, '  Star', ...
    'Color', [0.85 0.15 0.15], 'FontWeight', 'bold');
hNormal = quiver3(axMain, 0, 0, 0, 0, 0, 0, 0, ...
    'Color', [0.15 0.35 0.85], 'LineWidth', 2.2, 'MaxHeadSize', 0.5);
hNormalText = text(axMain, 0, 0, 0, '  Plane normal', ...
    'Color', [0.15 0.35 0.85], 'FontWeight', 'bold');
hAngleText = text(axMain, -0.95*data.mainLim, 0.88*data.mainLim, 0.88*data.mainLim, '', ...
    'FontSize', 11, 'BackgroundColor', 'w', 'Margin', 2);
hTimeText = text(axMain, -0.95*data.mainLim, -0.93*data.mainLim, 0.90*data.mainLim, '', ...
    'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 2);

% -------------------------------------------------------------------------
% OPD-relative panel
% -------------------------------------------------------------------------
hold(axOpd, 'on');
grid(axOpd, 'on');
xlabel(axOpd, data.timeLabel);
ylabel(axOpd, 'Relative OPD [m]');
title(axOpd, 'OPD Relative to the Mean');
xlim(axOpd, [data.tt(1), data.tt(end)]);
ylim(axOpd, data.opdYLim);

plot(axOpd, data.tt([1 end]), [0 0], 'k:', 'LineWidth', 1.0);
if isfinite(data.opdTol_m)
    plot(axOpd, data.tt([1 end]), [ data.opdTol_m  data.opdTol_m], '--', ...
        'Color', [0.55 0.55 0.55], 'LineWidth', 1.0);
    plot(axOpd, data.tt([1 end]), [-data.opdTol_m -data.opdTol_m], '--', ...
        'Color', [0.55 0.55 0.55], 'LineWidth', 1.0);
end

hOpdBg = gobjects(1, Nc);
hOpdActive = gobjects(1, Nc);
hOpdMarker = gobjects(1, Nc);
for j = 1:Nc
    hOpdBg(j) = plot(axOpd, data.tt, data.opdRel_m(:,j), '-', ...
        'Color', local_lighten_color(colors(j,:), 0.75), 'LineWidth', 0.9);
    hOpdActive(j) = plot(axOpd, nan, nan, '-', ...
        'Color', colors(j,:), 'LineWidth', 2.0);
    hOpdMarker(j) = plot(axOpd, nan, nan, 'o', ...
        'Color', colors(j,:), 'MarkerFaceColor', colors(j,:), 'MarkerSize', 6);
end
hOpdCursor = plot(axOpd, [data.tt(1) data.tt(1)], data.opdYLim, 'k--', 'LineWidth', 1.1);
plot(axOpd, [data.holdStartTimePlot data.holdStartTimePlot], data.opdYLim, ':', ...
    'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
legend(axOpd, hOpdActive, local_collector_labels(Nc), 'Location', 'northeast');

% -------------------------------------------------------------------------
% Control panels
% -------------------------------------------------------------------------
hCtrlActive = gobjects(Nc, 3);
hCtrlMarker = gobjects(Nc, 3);
hCtrlCursor = gobjects(1, Nc);

for j = 1:Nc
    hold(axCtrl(j), 'on');
    grid(axCtrl(j), 'on');
    xlim(axCtrl(j), [data.tt(1), data.tt(end)]);
    ylim(axCtrl(j), data.ctrlYLim);
    ylabel(axCtrl(j), 'u');
    title(axCtrl(j), sprintf('Collector %d RTN control', j));

    if j == Nc
        xlabel(axCtrl(j), data.timeLabel);
    else
        axCtrl(j).XTickLabel = [];
    end

    plot(axCtrl(j), data.tt([1 end]), [0 0], 'k:', 'LineWidth', 0.9);
    plot(axCtrl(j), [data.holdStartTimePlot data.holdStartTimePlot], data.ctrlYLim, ':', ...
        'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);

    for ctrlComp = 1:3
        plot(axCtrl(j), data.tt, data.Ufull{j}(:,ctrlComp), '-', ...
            'Color', local_lighten_color(componentColors(ctrlComp,:), 0.78), 'LineWidth', 0.8);
        hCtrlActive(j,ctrlComp) = plot(axCtrl(j), nan, nan, '-', ...
            'Color', componentColors(ctrlComp,:), 'LineWidth', 1.7);
        hCtrlMarker(j,ctrlComp) = plot(axCtrl(j), nan, nan, 'o', ...
            'Color', componentColors(ctrlComp,:), ...
            'MarkerFaceColor', componentColors(ctrlComp,:), ...
            'MarkerSize', 4.5);
    end

    hCtrlCursor(j) = plot(axCtrl(j), [data.tt(1) data.tt(1)], data.ctrlYLim, 'k--', 'LineWidth', 1.0);
end

legend(axCtrl(1), [hCtrlActive(1,1) hCtrlActive(1,2) hCtrlActive(1,3)], ...
    {'u_R','u_T','u_N'}, 'Location', 'northwest');

drawnow;

writerObj = [];
if ~isempty(opts.videoFile)
    writerObj = VideoWriter(opts.videoFile, 'MPEG-4');
    writerObj.FrameRate = opts.frameRate;
    open(writerObj);
end

setappdata(fig, 'cleanupWriter', onCleanup(@() local_close_writer(writerObj)));

for ii = 1:numel(frameList)
    k = frameList(ii);
    local_update_frame(k);

    drawnow;

    if ~isempty(writerObj)
        writeVideo(writerObj, getframe(fig));
    end

    if opts.pauseTime > 0
        pause(opts.pauseTime);
    end

    % When we hit the zoom trigger frame, break out of the coarse loop and
    % switch to the dense zoomed segment.
    if ii == zoomTriggerFrame
        break;
    end
end

% ---- Zoomed segment: snap axes, then animate on the dense grid -----------
if zoomTriggerFrame <= numel(frameList)

    % Snap zoom on all panels
    xlim(axOpd, data.zoomXLim);
    ylim(axOpd, data.zoomOpdYLim);
    for jz = 1:Nc
        xlim(axCtrl(jz), data.zoomXLim);
        ylim(axCtrl(jz), data.zoomCtrlYLim);
    end

    pauseZoom = opts.pauseTime * opts.ocpPauseMultiplier;

    for iz = 1:numel(data.ttZoom)
        local_update_frame_zoom(iz);
        drawnow;

        if ~isempty(writerObj)
            writeVideo(writerObj, getframe(fig));
        end

        if pauseZoom > 0
            pause(pauseZoom);
        end
    end
end

    function local_update_frame(k)
        P = zeros(Nc, 3);
        for jj = 1:Nc
            P(jj,:) = data.Rfull{jj}(k,:);
            k0 = max(1, k - opts.trailLength);
            set(hTrail(jj), ...
                'XData', data.Rfull{jj}(k0:k,1), ...
                'YData', data.Rfull{jj}(k0:k,2), ...
                'ZData', data.Rfull{jj}(k0:k,3));
            set(hCollector(jj), ...
                'XData', P(jj,1), ...
                'YData', P(jj,2), ...
                'ZData', P(jj,3));
        end

        xConn = [P(:,1); P(1,1)];
        yConn = [P(:,2); P(1,2)];
        zConn = [P(:,3); P(1,3)];
        set(hConn, 'XData', xConn, 'YData', yConn, 'ZData', zConn);

        % Centroid of the collector plane (used as the common origin for
        % both the star direction and the plane-normal arrows).
        c = mean(P(1:Nc,:), 1);

        svec = data.scaleStar * data.sRTN(k,:);
        set(hStar, 'XData', c(1), 'YData', c(2), 'ZData', c(3), ...
            'UData', svec(1), 'VData', svec(2), 'WData', svec(3));
        set(hStarText, 'Position', c + svec, 'String', '  Star');

        if Nc >= 3
            v1 = P(2,:) - P(1,:);
            v2 = P(3,:) - P(1,:);
            nvec = cross(v1, v2);
            nmag = norm(nvec);

            if nmag > 0
                nhat = nvec / nmag;
                nplot = data.scaleNormal * nhat;

                set(hNormal, ...
                    'XData', c(1), 'YData', c(2), 'ZData', c(3), ...
                    'UData', nplot(1), 'VData', nplot(2), 'WData', nplot(3), ...
                    'Visible', 'on');
                set(hNormalText, ...
                    'Position', c + nplot, ...
                    'String', '  Plane normal', ...
                    'Visible', 'on');

                if opts.showPlane
                    set(hPlane, ...
                        'XData', P(1:3,1), ...
                        'YData', P(1:3,2), ...
                        'ZData', P(1:3,3), ...
                        'Visible', 'on');
                else
                    set(hPlane, 'Visible', 'off');
                end

                alignAngle = acosd(max(-1, min(1, dot(nhat, data.sRTN(k,:)))));
                set(hAngleText, 'String', sprintf('\\angle(n_{plane}, s) = %.2f deg', alignAngle));
            else
                set(hNormal, 'Visible', 'off');
                set(hNormalText, 'Visible', 'off');
                set(hPlane, 'Visible', 'off');
                set(hAngleText, 'String', '');
            end
        end

        set(hTimeText, 'String', sprintf('t = %.2f %s', data.tt(k), data.timeUnitText));

        for jj = 1:Nc
            set(hOpdActive(jj), ...
                'XData', data.tt(1:k), ...
                'YData', data.opdRel_m(1:k,jj));
            set(hOpdMarker(jj), ...
                'XData', data.tt(k), ...
                'YData', data.opdRel_m(k,jj));
        end
        set(hOpdCursor, ...
            'XData', [data.tt(k) data.tt(k)], ...
            'YData', ylim(axOpd));

        for jj = 1:Nc
            for liveComp = 1:3
                set(hCtrlActive(jj,liveComp), ...
                    'XData', data.tt(1:k), ...
                    'YData', data.Ufull{jj}(1:k,liveComp));
                set(hCtrlMarker(jj,liveComp), ...
                    'XData', data.tt(k), ...
                    'YData', data.Ufull{jj}(k,liveComp));
            end
            set(hCtrlCursor(jj), ...
                'XData', [data.tt(k) data.tt(k)], ...
                'YData', ylim(axCtrl(jj)));
        end
    end
function local_update_frame_zoom(iz)
        P = zeros(Nc, 3);
        for jj = 1:Nc
            P(jj,:) = data.RfullZoom{jj}(iz,:);
            % trail: find how many dense steps correspond to trailLength base steps
            trailDense = opts.trailLength * 2;
            iz0 = max(1, iz - trailDense);
            set(hTrail(jj), ...
                'XData', data.RfullZoom{jj}(iz0:iz,1), ...
                'YData', data.RfullZoom{jj}(iz0:iz,2), ...
                'ZData', data.RfullZoom{jj}(iz0:iz,3));
            set(hCollector(jj), ...
                'XData', P(jj,1), ...
                'YData', P(jj,2), ...
                'ZData', P(jj,3));
        end

        xConn = [P(:,1); P(1,1)];
        yConn = [P(:,2); P(1,2)];
        zConn = [P(:,3); P(1,3)];
        set(hConn, 'XData', xConn, 'YData', yConn, 'ZData', zConn);

        c = mean(P(1:Nc,:), 1);

        svec = data.scaleStar * data.sRTNzoom(iz,:);
        set(hStar, 'XData', c(1), 'YData', c(2), 'ZData', c(3), ...
            'UData', svec(1), 'VData', svec(2), 'WData', svec(3));
        set(hStarText, 'Position', c + svec, 'String', '  Star');

        if Nc >= 3
            v1 = P(2,:) - P(1,:);
            v2 = P(3,:) - P(1,:);
            nvec = cross(v1, v2);
            nmag = norm(nvec);

            if nmag > 0
                nhat = nvec / nmag;
                nplot = data.scaleNormal * nhat;

                set(hNormal, ...
                    'XData', c(1), 'YData', c(2), 'ZData', c(3), ...
                    'UData', nplot(1), 'VData', nplot(2), 'WData', nplot(3), ...
                    'Visible', 'on');
                set(hNormalText, ...
                    'Position', c + nplot, ...
                    'String', '  Plane normal', ...
                    'Visible', 'on');

                if opts.showPlane
                    set(hPlane, ...
                        'XData', P(1:3,1), ...
                        'YData', P(1:3,2), ...
                        'ZData', P(1:3,3), ...
                        'Visible', 'on');
                else
                    set(hPlane, 'Visible', 'off');
                end

                alignAngle = acosd(max(-1, min(1, dot(nhat, data.sRTNzoom(iz,:)))));
                set(hAngleText, 'String', sprintf('\\angle(n_{plane}, s) = %.2f deg', alignAngle));
            else
                set(hNormal, 'Visible', 'off');
                set(hNormalText, 'Visible', 'off');
                set(hPlane, 'Visible', 'off');
                set(hAngleText, 'String', '');
            end
        end

        set(hTimeText, 'String', sprintf('t = %.2f %s', data.ttZoom(iz), data.timeUnitText));

        for jj = 1:Nc
            set(hOpdActive(jj), ...
                'XData', data.ttZoom(1:iz), ...
                'YData', data.opdRelZoom(1:iz,jj));
            set(hOpdMarker(jj), ...
                'XData', data.ttZoom(iz), ...
                'YData', data.opdRelZoom(iz,jj));
        end
        set(hOpdCursor, ...
            'XData', [data.ttZoom(iz) data.ttZoom(iz)], ...
            'YData', ylim(axOpd));

        for jj = 1:Nc
            for liveComp = 1:3
                set(hCtrlActive(jj,liveComp), ...
                    'XData', data.ttZoom(1:iz), ...
                    'YData', data.UfullZoom{jj}(1:iz,liveComp));
                set(hCtrlMarker(jj,liveComp), ...
                    'XData', data.ttZoom(iz), ...
                    'YData', data.UfullZoom{jj}(iz,liveComp));
            end
            set(hCtrlCursor(jj), ...
                'XData', [data.ttZoom(iz) data.ttZoom(iz)], ...
                'YData', ylim(axCtrl(jj)));
        end
    end

end

% -------------------------------------------------------------------------
function data = local_build_full_history(out, sol, kStart, mode, target, opts)

tBase = out.t(:);
solT = sol.t(:);
Nc = numel(out.dr_rtn);

if numel(tBase) < 2
    error('out.t must contain at least two samples.');
end
if numel(solT) < 2
    error('sol.t must contain at least two samples.');
end

if isfield(opts, 'dtSample_s') && ~isempty(opts.dtSample_s)
    dtSample = opts.dtSample_s;
else
    dtSample = median(diff(tBase));
end

tauHold = (0:dtSample:solT(end)).';
if abs(tauHold(end) - solT(end)) > 1e-9
    tauHold(end+1,1) = solT(end);
end

holdAbsT = tBase(kStart) + tauHold;
if kStart > 1
    tFull = [tBase(1:kStart-1); holdAbsT];
else
    tFull = holdAbsT;
end

chiefBase = local_extract_chief_history(out);
chiefHold = interp1(solT, sol.chief(:,1:6), tauHold, 'linear', 'extrap');
if kStart > 1
    chiefFull = [chiefBase(1:kStart-1,:); chiefHold];
else
    chiefFull = chiefHold;
end

Rfull = cell(1, Nc);
Ufull = cell(1, Nc);
for j = 1:Nc
    Rhold = interp1(solT, sol.RTN{j}, tauHold, 'linear', 'extrap');
    Uhold = local_interp_previous(solT(1:end-1), sol.U{j}, tauHold);

    if kStart > 1
        Rfull{j} = [out.dr_rtn{j}(1:kStart-1,:); Rhold];
        Ufull{j} = [zeros(kStart-1,3); Uhold];
    else
        Rfull{j} = Rhold;
        Ufull{j} = Uhold;
    end
end

chiefStruct = struct();
chiefStruct.u = chiefFull(:,6);
chiefStruct.inc = chiefFull(:,4);
chiefStruct.RAAN = chiefFull(:,5);

opdMat = zeros(numel(tFull), Nc);
for j = 1:Nc
    opdMat(:,j) = opd_from_rtn(Rfull{j}, mode, target, chiefStruct);
end

opdMean = mean(opdMat, 2);
opdRel_m = 1000 * (opdMat - opdMean);

[tt, timeLabel, timeUnitText] = local_scale_time(tFull, opts.timeUnit);

rmax = 0;
for j = 1:Nc
    rmax = max(rmax, max(vecnorm(Rfull{j}, 2, 2)));
end
if rmax <= 0
    rmax = 1;
end

ctrlMax = 0;
for j = 1:Nc
    ctrlMax = max(ctrlMax, max(abs(Ufull{j}(:))));
end
ctrlMax = max(ctrlMax, 1);

opdMax = max(abs(opdRel_m(:)));
opdTol_m = inf;
if isfield(sol, 'Dmax_m') && ~isempty(sol.Dmax_m)
    opdTol_m = sol.Dmax_m / 2;
elseif isfield(opts, 'Dmax_m') && ~isempty(opts.Dmax_m)
    opdTol_m = opts.Dmax_m / 2;
end
if isfinite(opdTol_m)
    opdMax = max(opdMax, opdTol_m);
end
if opdMax <= 0
    opdMax = 1;
end

sRTN = zeros(numel(tFull), 3);
for k = 1:numel(tFull)
    switch lower(mode)
        case 'phibeta'
            phi0 = target(1);
            beta = target(2);
            u = chiefFull(k,6);
            sRTN(k,:) = [cos(beta)*cos(phi0-u), ...
                         cos(beta)*sin(phi0-u), ...
                         sin(beta)];
        case 'radec'
            sRTN(k,:) = local_source_rtn_from_radec( ...
                target(1), target(2), chiefFull(k,4), chiefFull(k,5), chiefFull(k,6)).';
        otherwise
            error('Unknown target mode "%s".', mode);
    end
end

data = struct();
data.Nc = Nc;
data.Nt = numel(tFull);
data.t = tFull;
data.tt = tt;
data.timeLabel = timeLabel;
data.timeUnitText = timeUnitText;
data.holdStartTime = tBase(kStart);
data.holdStartTimePlot = local_scale_single_time(tBase(kStart), opts.timeUnit);
data.Rfull = Rfull;
data.Ufull = Ufull;
data.chiefFull = chiefFull;
data.sRTN = sRTN;
data.opdRel_m = opdRel_m;
data.mainLim = 1.5 * rmax;
data.scaleStar = 1.2 * rmax;
data.scaleNormal = 0.8 * rmax;
data.ctrlYLim = 1.10 * [-ctrlMax ctrlMax];
data.opdYLim = 1.10 * [-opdMax opdMax];
data.opdTol_m = opdTol_m;

% ---- OCP zoom limits (used once cursor enters the zoom lead window) ------
% X limits: from a small pad before the hold start to the end of the signal.
% The pad is 2% of the total time span so the hold-start marker stays visible.
tPad = 0.02 * (tt(end) - tt(1));
data.zoomXLim = [data.holdStartTimePlot - tPad, tt(end)];

% Y limits for OPD during OCP: auto-fit over the OCP portion of the signal.
ocpMask = tt >= data.holdStartTimePlot;
if any(ocpMask)
    opdOcpMax = max(abs(opdRel_m(ocpMask, :)), [], 'all');
else
    opdOcpMax = opdMax;
end
if isfinite(opdTol_m)
    opdOcpMax = max(opdOcpMax, opdTol_m);
end
if opdOcpMax <= 0
    opdOcpMax = opdMax;
end
data.zoomOpdYLim = 1.10 * [-opdOcpMax opdOcpMax];

% Y limits for each control channel during OCP.
ctrlOcpMax = zeros(1, Nc);
for j = 1:Nc
    v = Ufull{j}(ocpMask, :);
    if ~isempty(v)
        ctrlOcpMax(j) = max(abs(v(:)));
    else
        ctrlOcpMax(j) = ctrlMax;
    end
end
globalCtrlOcpMax = max(ctrlOcpMax);
if globalCtrlOcpMax <= 0
    globalCtrlOcpMax = ctrlMax;
end
data.zoomCtrlYLim = 1.10 * [-globalCtrlOcpMax globalCtrlOcpMax];
% ---- Dense interpolated grid for the smooth zoomed animation segment ----
% This covers from the zoom lead-in point to the end of the signal.
% The lead-in time is approximated from zoomLeadFrames * step * dtSample.
dtDense = dtSample / 4;   % 4x finer than the base sample rate
tZoomStart = max(tFull(1), tBase(kStart) - opts.zoomLeadFrames * opts.step * dtSample);
tZoomGrid  = (tZoomStart : dtDense : tFull(end)).';
if tZoomGrid(end) < tFull(end)
    tZoomGrid(end+1) = tFull(end);
end

ttZoom = local_scale_time_vec(tZoomGrid, opts.timeUnit);

% Interpolate all signals onto the dense grid
RfullZoom  = cell(1, Nc);
UfullZoom  = cell(1, Nc);
for j = 1:Nc
    RfullZoom{j} = interp1(tFull, Rfull{j},   tZoomGrid, 'linear', 'extrap');
    UfullZoom{j} = interp1(tFull, Ufull{j},   tZoomGrid, 'linear', 'extrap');
end
opdRelZoom = interp1(tFull, opdRel_m, tZoomGrid, 'linear', 'extrap');
sRTNzoom   = interp1(tFull, sRTN,    tZoomGrid, 'linear', 'extrap');
% Re-normalise sRTN rows after interpolation
rowNorms = max(vecnorm(sRTNzoom, 2, 2), 1e-12);
sRTNzoom = sRTNzoom ./ rowNorms;

chiefZoom = interp1(tFull, chiefFull, tZoomGrid, 'linear', 'extrap');

data.tZoomGrid   = tZoomGrid;
data.ttZoom      = ttZoom;
data.RfullZoom   = RfullZoom;
data.UfullZoom   = UfullZoom;
data.opdRelZoom  = opdRelZoom;
data.sRTNzoom    = sRTNzoom;
data.chiefZoom   = chiefZoom;
end

% -------------------------------------------------------------------------
function chiefHist = local_extract_chief_history(out)

if isfield(out, 'states') && isfield(out.states, 'chief')
    chiefHist = out.states.chief(:,1:6);
elseif isfield(out, 'chief')
    chiefHist = out.chief(:,1:6);
else
    error('Could not find chief state history in out.');
end

end

% -------------------------------------------------------------------------
function [mode, target] = local_resolve_target(out, opts)

if isfield(opts, 'mode') && ~isempty(opts.mode)
    mode = opts.mode;
else
    if isfield(out, 'star') && isfield(out.star, 'phi0') && isfield(out.star, 'beta')
        mode = 'phibeta';
    else
        error('Could not infer target mode. Provide opts.mode and opts.target.');
    end
end

if isfield(opts, 'target') && ~isempty(opts.target)
    target = opts.target(:).';
    return;
end

switch lower(mode)
    case 'phibeta'
        if isfield(out, 'star') && isfield(out.star, 'phi0') && isfield(out.star, 'beta')
            target = [out.star.phi0, out.star.beta];
        else
            error('For phibeta mode, target must be provided or available in out.star.');
        end
    case 'radec'
        if isfield(out, 'star') && isfield(out.star, 'ra') && isfield(out.star, 'dec')
            target = [out.star.ra, out.star.dec];
        else
            error('For radec mode, target must be provided or available in out.star.');
        end
    otherwise
        error('Unknown target mode "%s".', mode);
end

end

% -------------------------------------------------------------------------
function values = local_interp_previous(x, y, xi)

x = x(:);
xi = xi(:);

if isempty(x)
    values = zeros(numel(xi), size(y,2));
    return;
end

if isscalar(x)
    values = repmat(y(1,:), numel(xi), 1);
    return;
end

values = interp1(x, y, xi, 'previous', 'extrap');

end

% -------------------------------------------------------------------------
function [tt, label, unitText] = local_scale_time(t, timeUnit)

switch lower(timeUnit)
    case 's'
        tt = t;
        label = 'Time [s]';
        unitText = 's';
    case 'min'
        tt = t / 60;
        label = 'Time [min]';
        unitText = 'min';
    case 'hr'
        tt = t / 3600;
        label = 'Time [hr]';
        unitText = 'hr';
    otherwise
        error('Unknown opts.timeUnit "%s".', timeUnit);
end

end

% -------------------------------------------------------------------------
function tScaled = local_scale_single_time(t, timeUnit)

switch lower(timeUnit)
    case 's'
        tScaled = t;
    case 'min'
        tScaled = t / 60;
    case 'hr'
        tScaled = t / 3600;
    otherwise
        error('Unknown opts.timeUnit "%s".', timeUnit);
end

end

% -------------------------------------------------------------------------
function ttVec = local_scale_time_vec(t, timeUnit)

switch lower(timeUnit)
    case 's',   ttVec = t;
    case 'min', ttVec = t / 60;
    case 'hr',  ttVec = t / 3600;
    otherwise,  error('Unknown opts.timeUnit "%s".', timeUnit);
end

end

% -------------------------------------------------------------------------
function labels = local_collector_labels(Nc)

labels = cell(1, Nc);
for j = 1:Nc
    labels{j} = sprintf('Collector %d', j);
end

end

% -------------------------------------------------------------------------
function cOut = local_lighten_color(cIn, amount)

amount = min(max(amount, 0), 1);
cOut = cIn + amount * (1 - cIn);

end

% -------------------------------------------------------------------------
function local_close_writer(writerObj)
if ~isempty(writerObj)
    close(writerObj);
end
end

% -------------------------------------------------------------------------
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

Nhat = [ sO*si;
        -cO*si;
         ci ];

sRTN = [dot(sI,Rhat);
        dot(sI,That);
        dot(sI,Nhat)];
end