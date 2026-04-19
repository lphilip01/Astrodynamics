function animate_formation_star_plane(t, out, chief, target, mode, opts)
% animate_formation_star_plane
%
% Interactive animation of 3D RTN collector positions with:
%   - star direction arrow
%   - collector-plane normal arrow
%   - optional collector plane patch
%
% Controls:
%   - Play / Pause
%   - Speed +
%   - Speed -
%   - Forward / Reverse
%
% Inputs:
%   t      : Nx1 time vector [s]
%   out    : output struct from opd_pipeline_from_qns (multi-deputy)
%   chief  : struct with chief geometry
%            for mode='phibeta': chief.u
%            for mode='radec'  : chief.u, chief.inc, chief.RAAN
%   target : [phi0 beta] or [ra dec]
%   mode   : 'phibeta' or 'radec'
%   opts   : optional struct
%            opts.step           default 20
%            opts.scaleStar      default auto
%            opts.scaleNormal    default auto
%            opts.showPlane      default true
%            opts.axisEqual      default true
%            opts.trailLength    default 0
%            opts.pauseTime      default 0.05
%            opts.speedFactor    default 1.5
%            opts.minPauseTime   default 0.01
%
% Notes:
%   - Requires at least 3 collectors to define a plane.
%   - Uses out.dr_rtn{j}, each [Nt x 3], in km.

if nargin < 6
    opts = struct();
end

if ~isfield(opts,'step'),         opts.step = 5; end
if ~isfield(opts,'showPlane'),    opts.showPlane = true; end
if ~isfield(opts,'axisEqual'),    opts.axisEqual = true; end
if ~isfield(opts,'trailLength'),  opts.trailLength = 0; end
if ~isfield(opts,'pauseTime'),    opts.pauseTime = 0.001; end
if ~isfield(opts,'speedFactor'),  opts.speedFactor = 1.5; end
if ~isfield(opts,'minPauseTime'), opts.minPauseTime = 0.001; end

opts.step = max(1, round(opts.step));

if ~iscell(out.dr_rtn)
    error('out.dr_rtn must be a cell array with at least 3 collectors.');
end

Nc = numel(out.dr_rtn);
if Nc < 3
    error('Need at least 3 collectors to define a plane.');
end

Nt = length(t);

% Build collector position array: Nt x 3 x Nc
R = zeros(Nt,3,Nc);
for j = 1:Nc
    R(:,:,j) = out.dr_rtn{j};
end

% Determine scaling from formation size
rmax = 0;
for j = 1:Nc
    rmax = max(rmax, max(vecnorm(R(:,:,j),2,2)));
end

if ~isfield(opts,'scaleStar'),   opts.scaleStar = 1.2*rmax; end
if ~isfield(opts,'scaleNormal'), opts.scaleNormal = 0.8*rmax; end

% Precompute source vector in RTN
sRTN = zeros(Nt,3);
for k = 1:Nt
    switch lower(mode)
        case 'phibeta'
            phi0 = target(1);
            beta = target(2);
            u = chief.u(k);

            sRTN(k,:) = [cos(beta)*cos(phi0-u), ...
                         cos(beta)*sin(phi0-u), ...
                         sin(beta)];

        case 'radec'
            ra  = target(1);
            dec = target(2);

            inc  = chief.inc(k);
            RAAN = chief.RAAN(k);
            u    = chief.u(k);

            sRTN(k,:) = local_source_rtn_from_radec(ra, dec, inc, RAAN, u).';

        otherwise
            error('Unknown mode. Use ''phibeta'' or ''radec''.');
    end
end

lim = 1.5*rmax;
if lim == 0
    lim = 1;
end

frameList = 1:opts.step:Nt;
if frameList(end) ~= Nt
    frameList(end+1) = Nt;
end
nFrames = numel(frameList);

colors = lines(Nc);

% Playback state
state.idx        = 1;
state.isPlaying  = false;
state.direction  = 1;    % 1 = forward, -1 = reverse
state.speed      = 1.0;  % playback multiplier
state.loopActive = false;

% Figure and axes
fig = figure( ...
    'Color','w', ...
    'Name','Interactive Formation Animation', ...
    'NumberTitle','off', ...
    'Units','normalized', ...
    'Position',[0.15 0.12 0.7 0.75]);

ax = axes('Parent', fig, 'Position', [0.08 0.22 0.86 0.72]);

% UI controls
btnPlay = uicontrol(fig, 'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.12 0.08 0.12 0.07], ...
    'String','Play', ...
    'FontSize',10, ...
    'Callback',@togglePlay);

btnSlow = uicontrol(fig, 'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.28 0.08 0.12 0.07], ...
    'String','Speed -', ...
    'FontSize',10, ...
    'Callback',@slowDown);

btnFast = uicontrol(fig, 'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.44 0.08 0.12 0.07], ...
    'String','Speed +', ...
    'FontSize',10, ...
    'Callback',@speedUp);

btnDir = uicontrol(fig, 'Style','pushbutton', ...
    'Units','normalized', ...
    'Position',[0.60 0.08 0.14 0.07], ...
    'String','Direction: Forward', ...
    'FontSize',10, ...
    'Callback',@toggleDirection);

txtInfo = uicontrol(fig, 'Style','text', ...
    'Units','normalized', ...
    'Position',[0.77 0.075 0.18 0.08], ...
    'String','Speed: 1.00x', ...
    'BackgroundColor','w', ...
    'FontSize',10, ...
    'HorizontalAlignment','left');

guidata(fig, state);
renderFrame();
updateControls();

    function togglePlay(~,~)
        if ~ishandle(fig), return; end
        state = guidata(fig);
        state.isPlaying = ~state.isPlaying;
        guidata(fig, state);
        updateControls();

        if state.isPlaying
            playbackLoop();
        end
    end

    function slowDown(~,~)
        if ~ishandle(fig), return; end
        state = guidata(fig);
        state.speed = max(0.25, state.speed / opts.speedFactor);
        guidata(fig, state);
        updateControls();
    end

    function speedUp(~,~)
        if ~ishandle(fig), return; end
        state = guidata(fig);
        state.speed = min(16.0, state.speed * opts.speedFactor);
        guidata(fig, state);
        updateControls();
    end

    function toggleDirection(~,~)
        if ~ishandle(fig), return; end
        state = guidata(fig);
        state.direction = -state.direction;
        guidata(fig, state);
        updateControls();
    end

    function updateControls()
        if ~ishandle(fig), return; end
        state = guidata(fig);

        if state.isPlaying
            btnPlay.String = 'Pause';
        else
            btnPlay.String = 'Play';
        end

        if state.direction > 0
            dirText = 'Forward';
        else
            dirText = 'Reverse';
        end

        btnDir.String = ['Direction: ' dirText];
        txtInfo.String = sprintf('Speed: %.2fx\nFrame: %d / %d', ...
            state.speed, state.idx, nFrames);
    end

    function playbackLoop()
        if ~ishandle(fig), return; end

        state = guidata(fig);
        if state.loopActive
            return;
        end
        state.loopActive = true;
        guidata(fig, state);

        while ishandle(fig)
            state = guidata(fig);
            if ~state.isPlaying
                break;
            end

            renderFrame();
            advanceFrame();

            pause(max(opts.minPauseTime, opts.pauseTime / state.speed));
            drawnow;
        end

        if ishandle(fig)
            state = guidata(fig);
            state.loopActive = false;
            guidata(fig, state);
            updateControls();
        end
    end

    function advanceFrame()
        if ~ishandle(fig), return; end
        state = guidata(fig);

        state.idx = state.idx + state.direction;

        if state.idx > nFrames
            state.idx = 1;
        elseif state.idx < 1
            state.idx = nFrames;
        end

        guidata(fig, state);
        updateControls();
    end

    function renderFrame()
        if ~ishandle(fig), return; end
        state = guidata(fig);

        k = frameList(state.idx);

        cla(ax);
        hold(ax, 'on');
        grid(ax, 'on');
        xlabel(ax, 'R [km]');
        ylabel(ax, 'T [km]');
        zlabel(ax, 'N [km]');
        title(ax, sprintf('Collectors / Star / Plane Normal   t = %.2f hr', t(k)/3600));
        view(ax, 3);

        if opts.axisEqual
            axis(ax, 'equal');
        end
        xlim(ax, [-lim lim]);
        ylim(ax, [-lim lim]);
        zlim(ax, [-lim lim]);

        % Combiner
        plot3(ax, 0,0,0,'ks','MarkerFaceColor','k','MarkerSize',8);

        % Collectors
        P = zeros(Nc,3);
        for j = 1:Nc
            P(j,:) = R(k,:,j);
            plot3(ax, P(j,1), P(j,2), P(j,3), 'o', ...
                'Color', colors(j,:), ...
                'MarkerFaceColor', colors(j,:), ...
                'MarkerSize', 8);

            if opts.trailLength > 0
                k1 = max(1, k-opts.trailLength);
                plot3(ax, R(k1:k,1,j), R(k1:k,2,j), R(k1:k,3,j), ...
                    '-', 'Color', colors(j,:), 'LineWidth', 1.0);
            end
        end

        % Connect collectors
        plot3(ax, P(:,1), P(:,2), P(:,3), 'k--', 'LineWidth', 1.0);
        plot3(ax, [P(end,1) P(1,1)], [P(end,2) P(1,2)], [P(end,3) P(1,3)], ...
            'k--', 'LineWidth', 1.0);

        % Star arrow
        svec = opts.scaleStar * sRTN(k,:);
        quiver3(ax, 0,0,0, svec(1), svec(2), svec(3), 0, ...
            'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
        text(ax, svec(1), svec(2), svec(3), '  Star', ...
            'Color','r', 'FontWeight','bold');

        % Plane normal from first 3 collectors
        v1 = P(2,:) - P(1,:);
        v2 = P(3,:) - P(1,:);
        nvec = cross(v1, v2);
        nmag = norm(nvec);

        if nmag > 0
            nhat = nvec / nmag;
            c = mean(P(1:3,:),1);

            nplot = opts.scaleNormal * nhat;
            quiver3(ax, c(1), c(2), c(3), nplot(1), nplot(2), nplot(3), 0, ...
                'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
            text(ax, c(1)+nplot(1), c(2)+nplot(2), c(3)+nplot(3), ...
                '  Plane normal', 'Color','b', 'FontWeight','bold');

            if opts.showPlane
                patch(ax, P(1:3,1), P(1:3,2), P(1:3,3), ...
                    [0.7 0.8 1.0], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
            end

            alignAngle = acosd(max(-1,min(1,dot(nhat, sRTN(k,:)))));
            text(ax, -0.95*lim, 0.9*lim, 0.9*lim, ...
                sprintf('\\angle(n_{plane}, s) = %.2f deg', alignAngle), ...
                'FontSize', 11, 'BackgroundColor', 'w');
        end

        drawnow;
        updateControls();
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
