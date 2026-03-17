function main()
%MAIN  Entry point for the GEO ROE Formation Explorer.
%
% Opens a screen-adaptive MATLAB figure with 8 synchronised visualisation
% panels and a slider control panel.  Drag sliders to update all plots in
% real time and build physical intuition for how each ROE component shapes
% the relative trajectory in the RTN frame.
%
% USAGE
%   cd geo_roe_explorer
%   main()
%
% LAYOUT
%   Left 228 px  : control panel (orbit selector, presets, 9 sliders,
%                  UV toggle, formation metrics readout)
%   Right region : 4 rows × 2 columns of synchronised plot panels
%
% REFERENCE
%   D'Amico & Montenbruck (2006) JGCD; Rizza et al. (2026, STARI).

    %% ── Add presets folder to path ──────────────────────────────────────
    tool_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(tool_dir, 'presets'));

    %% ── Unicode glyphs for readable slider labels ───────────────────────
    %  uicontrol text widgets accept Unicode code-points directly and render
    %  them without needing a TeX/LaTeX interpreter (which is unreliable
    %  for uicontrol across platforms).
    gd  = char(948);    % δ  Greek small delta
    gl  = char(955);    % λ  Greek small lambda
    gb  = char(946);    % β  Greek small beta
    gph = char(966);    % φ  Greek small phi
    cdt = char(183);    % ·  Middle dot  (for "a·δ" notation)
    deg = char(176);    % °  Degree sign
    mdash = char(8212); % —  Em dash

    %% ── Screen-adaptive figure geometry ────────────────────────────────
    scr = get(0, 'ScreenSize');   % [1  1  screen_width  screen_height]  px
    fw  = min(1280, scr(3) - 30);
    fh  = min(710,  scr(4) - 90);   % 90 px headroom for taskbar + title bar
    fx  = max(5,  floor((scr(3) - fw) / 2));
    fy  = max(50, floor((scr(4) - fh) / 2));

    fig = figure( ...
        'Name',        ['ROE Formation Explorer  ' mdash ...
                        '  Philip Lopez + Claude'], ...
        'NumberTitle', 'off', ...
        'Position',    [fx, fy, fw, fh], ...
        'MenuBar',     'figure', ...
        'ToolBar',     'none', ...
        'Color',       [0.95, 0.95, 0.95], ...
        'CloseRequestFcn', @on_close, ...
        'Resize',      'off');

    %% ── Left control panel ──────────────────────────────────────────────
    %  Fixed 228-px-wide uipanel; all child controls use pixel units so
    %  positions are exact and nothing overlaps.
    ctrl_w = 228;
    ctrl = uipanel('Parent',          fig, ...
                   'Units',           'pixels', ...
                   'Position',        [0, 0, ctrl_w, fh], ...
                   'BackgroundColor', [0.92, 0.92, 0.96], ...
                   'BorderType',      'etchedin');

    %% ── Section: title bar ──────────────────────────────────────────────
    uicontrol(ctrl, 'Style','text', 'Units','pixels', ...
              'Position', [2, fh-29, ctrl_w-4, 26], ...
              'String',  'ROE Formation Explorer', ...
              'FontSize', 9, 'FontWeight','bold', ...
              'BackgroundColor', [0.18, 0.28, 0.58], ...
              'ForegroundColor', 'white', ...
              'HorizontalAlignment', 'center');

    %% ── Section: orbit type ─────────────────────────────────────────────
    orbit_bg = uibuttongroup(ctrl, ...
        'Units',    'pixels', ...
        'Position', [2, fh-68, ctrl_w-4, 37], ...
        'Title',    'Orbit Type', 'FontSize', 8, ...
        'BackgroundColor', [0.92, 0.92, 0.96], ...
        'SelectionChangedFcn', @on_orbit_changed);
    rb_geo = uicontrol(orbit_bg, 'Style','radiobutton', 'String','GEO', ...
                       'Units','pixels', 'Position',[4, 4, 92, 20], ...
                       'FontSize', 8, 'BackgroundColor', [0.92,0.92,0.96], ...
                       'Value', 1);
    rb_leo = uicontrol(orbit_bg, 'Style','radiobutton', 'String','LEO SSO', ...
                       'Units','pixels', 'Position',[100, 4, 116, 20], ...
                       'FontSize', 8, 'BackgroundColor', [0.92,0.92,0.96], ...
                       'Value', 0);

    %% ── Section: preset selector ────────────────────────────────────────
    uicontrol(ctrl, 'Style','text', 'Units','pixels', ...
              'Position',[4, fh-88, ctrl_w-8, 17], ...
              'String','ROE Preset:', 'FontSize',8, 'FontWeight','bold', ...
              'HorizontalAlignment','left', ...
              'BackgroundColor',[0.92,0.92,0.96]);

    preset_names = {
        ['1  ' gd 'a=0, ' gd gl '=500 m  (along-track only)'], ...
        ['2  ' gd 'ex=50 m  (2:1 CW ellipse)'], ...
        ['3  ' gd 'iy=100 m  (cross-track only)'], ...
        ['4  STARI science orbit'], ...
        ['5  E/I separated  (passively safe)'], ...
        ['6  Drifting  (' gd 'a' char(8800) '0)'], ...
        ['7  GEO Deneb  (' gph '0=90' deg ', ' gb '=45.3' deg ')']};
    dd_preset = uicontrol(ctrl, 'Style','popupmenu', 'Units','pixels', ...
                          'Position',[75, fh-93, ctrl_w-80, 24], ...
                          'String', preset_names, 'FontSize', 7, ...
                          'Callback', @on_preset_selected);

    %% ── Section: sliders ────────────────────────────────────────────────
    %  Each slider block occupies 40 px (18 px label row + 18 px slider + 4 px gap).
    %  Block 1 starts at sl_y0 (bottom of its label = sl_y0 + 20).
    %  No overlap with the preset popup whose top edge is at fh-114+24 = fh-90.
    %  sl_y0 = fh-90 - 4(gap) - 36(block internal) = fh-130  → label top = fh-110, fine.

    sl_y0  = fh - 130;   % bottom of slider block 1
    sl_blk = 40;         % px per block

    %  Slider definitions: {display_label, unit_str, min, max, default}
    sl_defs = {
        ['a' cdt gd 'a'],               'm',   -100,  100,    0;
        ['a' cdt gd gl],                'm',  -2000, 2000,  500;
        ['a' cdt gd 'ex'],              'm',   -200,  200,    0;
        ['a' cdt gd 'ey'],              'm',   -200,  200,    0;
        ['a' cdt gd 'ix'],              'm',   -200,  200,    0;
        ['a' cdt gd 'iy'],              'm',   -200,  200,  100;
        'Propagation',                  'orb',  0.5,   10,    3;
        [gb '  (star elev)'],           'deg',    5,   90,   45;
        [gph '0  (star az)'],           'deg',    0,  360,   90;
    };
    N_sl = size(sl_defs, 1);   % = 9

    sl_h = cell(N_sl, 1);   % slider handles
    vl_h = cell(N_sl, 1);   % value readout handles

    for k = 1:N_sl
        yb = sl_y0 - (k-1)*sl_blk;   % bottom pixel of this block

        % Parameter name (left side)
        uicontrol(ctrl, 'Style','text', 'Units','pixels', ...
                  'Position',[4, yb+21, 152, 16], ...
                  'String',[sl_defs{k,1} '  [' sl_defs{k,2} ']:'], ...
                  'FontSize', 7.5, 'HorizontalAlignment','left', ...
                  'BackgroundColor',[0.92,0.92,0.96]);

        % Current value (right side, blue tint background)
        vl_h{k} = uicontrol(ctrl, 'Style','text', 'Units','pixels', ...
                             'Position',[154, yb+20, 70, 18], ...
                             'String', fmt_val(sl_defs{k,5}, k), ...
                             'FontSize', 7.5, 'FontWeight','bold', ...
                             'BackgroundColor',[0.86,0.91,0.98], ...
                             'HorizontalAlignment','center');

        % Slider widget
        sl_h{k} = uicontrol(ctrl, 'Style','slider', 'Units','pixels', ...
                             'Position',[4, yb+2, ctrl_w-8, 17], ...
                             'Min', sl_defs{k,3}, ...
                             'Max', sl_defs{k,4}, ...
                             'Value', sl_defs{k,5}, ...
                             'SliderStep',[0.01, 0.05], ...
                             'Callback', @on_slider_changed);
        % Fires during drag (not only on button-up)
        addlistener(sl_h{k}, 'Value', 'PostSet', @on_slider_changed);
    end

    %% ── Section: UV toggle checkbox ─────────────────────────────────────
    y_chk = sl_y0 - N_sl*sl_blk + 4;
    chk_uv = uicontrol(ctrl, 'Style','checkbox', 'Units','pixels', ...
                        'Position',[4, y_chk, ctrl_w-8, 22], ...
                        'String','Show UV plane', 'Value', 1, ...
                        'FontSize', 8, 'BackgroundColor',[0.92,0.92,0.96], ...
                        'Callback', @on_slider_changed);

    %% ── Section: formation metrics readout ──────────────────────────────
    y_mhdr = y_chk - 28;
    uicontrol(ctrl, 'Style','text', 'Units','pixels', ...
              'Position',[2, y_mhdr, ctrl_w-4, 18], ...
              'String',[mdash mdash '  Formation Metrics  ' mdash mdash], ...
              'FontSize', 8, 'FontWeight','bold', ...
              'BackgroundColor',[0.92,0.92,0.96], ...
              'HorizontalAlignment','center');
    % Text fills all remaining space down to 4 px from panel bottom
    txt_metrics = uicontrol(ctrl, 'Style','text', 'Units','pixels', ...
                             'Position',[2, 4, ctrl_w-4, max(10, y_mhdr-8)], ...
                             'String','Computing...', ...
                             'FontSize', 7.5, 'HorizontalAlignment','left', ...
                             'BackgroundColor',[0.87,0.94,0.87]);

    %% ── 8 plot axes (4 rows × 2 cols) in the remaining figure area ──────
    axes_cell = make_axes(fig, fw, fh, ctrl_w);

    %% ── Load orbit presets ──────────────────────────────────────────────
    [chief_geo, ~] = geo_default();
    [chief_leo, ~] = leo_sso();

    %% ── Build GUI state struct and store in figure UserData ─────────────
    state.chief_geo   = chief_geo;
    state.chief_leo   = chief_leo;
    state.use_geo     = true;
    state.sl_h        = sl_h;
    state.vl_h        = vl_h;
    state.chk_uv      = chk_uv;
    state.dd_preset   = dd_preset;
    state.rb_geo      = rb_geo;
    state.rb_leo      = rb_leo;
    state.txt_metrics = txt_metrics;
    state.axes_cell   = axes_cell;
    state.ph          = [];
    state.initialized = false;
    % SRP perturbation struct (disabled; GVE integration hook)
    state.pert.srp_enabled       = false;
    state.pert.C_R               = 1.5;
    state.pert.AmR_chief         = 0.02;
    state.pert.AmR_deputy        = 0.0202;
    state.pert.sun_direction_ECI = [1;0;0];
    fig.UserData = state;

    %% ── First propagation and plot ──────────────────────────────────────
    do_update(fig);

    fprintf('\n==========================================================\n');
    fprintf('  GEO ROE Formation Explorer\n');
    fprintf('  Reference: D''Amico & Montenbruck (2006) JGCD\n');
    fprintf('  Drag sliders to explore relative orbit geometry in RTN.\n');
    fprintf('==========================================================\n\n');

    %% ==================================================================
    %%  NESTED CALLBACKS  (capture fig and slider handles via closure)
    %% ==================================================================

    function on_slider_changed(~, ~)
        do_update(fig);
    end

    function on_orbit_changed(~, ~)
        s = fig.UserData;
        s.use_geo = (s.rb_geo.Value == 1);
        fig.UserData = s;
        do_update(fig);
    end

    function on_preset_selected(src, ~)
        load_preset(fig, src.Value);
        do_update(fig);
    end

    function on_close(src, ~)
        delete(src);
    end

end   % ── end main ────────────────────────────────────────────────────────

% ===========================================================================
%%  HELPER: make_axes
% ===========================================================================
function axes_cell = make_axes(fig, fw, fh, ctrl_w)
%MAKE_AXES  Compute and create a 4×2 grid of axes in the plot area.
%
%  Positions are derived analytically from figure dimensions so the grid
%  fills available space with uniform margins and gaps.

    L_mar = 45;   R_mar = 12;   col_gap = 40;   % horizontal spacing [px]
    T_mar = 14;   B_mar = 30;   row_gap = 45;   % vertical spacing [px]

    plot_w  = fw - ctrl_w;
    pw_px   = (plot_w - L_mar - R_mar - col_gap)   / 2;   % panel width [px]
    ph_px   = (fh     - T_mar - B_mar - 3*row_gap) / 4;   % panel height [px]

    pw_n = pw_px / fw;   % normalised panel width
    ph_n = ph_px / fh;   % normalised panel height

    % Normalised left edges of the two columns
    c1 = (ctrl_w + L_mar) / fw;
    c2 = (ctrl_w + L_mar + pw_px + col_gap) / fw;

    % Normalised bottom edges of the four rows (top row first)
    rb = zeros(1, 4);
    rb(1) = (fh - T_mar - ph_px) / fh;
    for r = 2:4
        rb(r) = rb(r-1) - (ph_px + row_gap) / fh;
    end

    % Panel order: 1=3D(r1c1), 2=R-T(r1c2), 3=T-N(r2c1), 4=R-N(r2c2),
    %              5=ROEhist(r3c1), 6=e-space(r3c2), 7=i-space(r4c1), 8=UV(r4c2)
    cols = [c1; c2; c1; c2; c1; c2; c1; c2];
    rows = rb([1;1;2;2;3;3;4;4])';

    axes_cell = cell(8, 1);
    for k = 1:8
        axes_cell{k} = axes('Parent', fig, ...
                             'Units',    'normalized', ...
                             'Position', [cols(k), rows(k), pw_n, ph_n], ...
                             'FontSize',  7.5, ...
                             'Box',       'on');
    end
end

% ===========================================================================
%%  HELPER: do_update
% ===========================================================================
function do_update(fig)
%DO_UPDATE  Master update: read sliders → propagate → plot → show metrics.
%
%  Called on every slider PostSet event and on preset/orbit changes.
%  Uses plot_formation('update', ...) on subsequent calls to avoid clearing
%  axes (much faster than 'init'); only the first call uses 'init'.

    state = fig.UserData;
    if isempty(state), return; end

    % ── Read slider values ───────────────────────────────────────────────
    da       = state.sl_h{1}.Value;
    dlambda  = state.sl_h{2}.Value;
    dex      = state.sl_h{3}.Value;
    dey      = state.sl_h{4}.Value;
    dix      = state.sl_h{5}.Value;
    diy      = state.sl_h{6}.Value;
    N_orbits = state.sl_h{7}.Value;
    beta     = state.sl_h{8}.Value;
    phi_0    = state.sl_h{9}.Value;
    show_uv  = logical(state.chk_uv.Value);

    % ── Refresh value readouts ───────────────────────────────────────────
    vals = {da, dlambda, dex, dey, dix, diy, N_orbits, beta, phi_0};
    for k = 1:9
        state.vl_h{k}.String = fmt_val(vals{k}, k);
    end

    % ── Select chief orbit (GEO or LEO SSO) ─────────────────────────────
    if state.use_geo
        chief = state.chief_geo;
    else
        chief = state.chief_leo;
    end

    % ── Assemble ROE initial conditions [m] ─────────────────────────────
    roe0.da      = da;
    roe0.dlambda = dlambda;
    roe0.dex     = dex;
    roe0.dey     = dey;
    roe0.dix     = dix;
    roe0.diy     = diy;

    % ── Keplerian ROE propagation ────────────────────────────────────────
    [t_vec, roe_traj, u_c_vec] = propagate_roe(roe0, chief, N_orbits, 360);
    n        = sqrt(chief.mu / chief.a^3);
    t_orbits = t_vec * n / (2*pi);

    % ── RTN relative positions (linear mapping) ──────────────────────────
    rtn = roe_to_rtn(roe_traj, u_c_vec);

    % ── UV / OPD computation ─────────────────────────────────────────────
    uv = []; opd = [];
    if show_uv
        [uv, opd] = compute_uv_track(rtn, phi_0, beta);
    end

    % ── Formation metrics ────────────────────────────────────────────────
    metrics = compute_metrics(rtn, uv, opd, roe_traj);

    % ── Build data struct for plot_formation ────────────────────────────
    data.rtn       = rtn;
    data.roe_traj  = roe_traj;
    data.t_orbits  = t_orbits;
    data.uv        = uv;
    data.opd       = opd;
    data.metrics   = metrics;
    data.show_uv   = show_uv;
    data.star_phi0 = phi_0;
    data.star_beta  = beta;

    % ── Create or update all 8 panels ───────────────────────────────────
    if ~state.initialized
        ph = plot_formation('init', state.axes_cell, data, []);
        state.ph          = ph;
        state.initialized = true;
        drawnow;              % full render on first call
    else
        plot_formation('update', state.axes_cell, data, state.ph);
        drawnow limitrate;    % throttle during slider drag (~25 fps)
    end

    % ── Update GUI metrics text ──────────────────────────────────────────
    update_metrics_panel(state.txt_metrics, roe0, metrics, chief, ...
                         show_uv, phi_0, beta);

    fig.UserData = state;
end

% ===========================================================================
%%  HELPER: fmt_val
% ===========================================================================
function s = fmt_val(v, idx)
%FMT_VAL  Format a slider value for the compact readout label.
%  idx 1–6: ROE components [m], idx 7: orbits, idx 8–9: angles [deg]

    deg = char(176);
    if idx <= 6
        s = sprintf('%.0f m', v);
    elseif idx == 7
        s = sprintf('%.1f orb', v);
    else
        s = sprintf('%.0f%s', v, deg);
    end
end

% ===========================================================================
%%  HELPER: load_preset
% ===========================================================================
function load_preset(fig, idx)
%LOAD_PRESET  Push a named ROE configuration into the slider widgets.
%
%  Preset ROE values (absolute, meters) with physical interpretations:
%
%  1  Pure along-track  — da=0, dλ=500 m
%     Deputy oscillates purely along T-axis; no R or N motion.
%
%  2  2:1 CW ellipse    — da=0, dex=50 m
%     Classic Clohessy-Wiltshire 2:1 ellipse in R-T plane.
%
%  3  Cross-track only  — da=0, diy=100 m
%     Pure sinusoidal N motion; no in-plane excursion.
%
%  4  STARI science     — da=0, dλ=100 m, diy=100 m
%     T-N oscillation perpendicular to T-direction star (phi_0=90°).
%
%  5  E/I separated     — da=0, dex=50 m, dix=50 m  (de ∥ di)
%     Passively safe standby; bounded 3-D relative orbit.
%
%  6  Drifting          — da=1 m, dλ=500 m
%     Secular along-track drift; growing separation with time.
%
%  7  GEO Deneb target  — phi_0=90°, beta=45.3° (Deneb declination)
%     T-N ellipse perpendicular to Deneb direction from GEO.

    state = fig.UserData;

    % Row: [da, dlambda, dex, dey, dix, diy, N_orbits, beta_deg, phi_0_deg]
    switch idx
        case 1,  v = [0,   500,  0,  0,    0,    0,  3, 45,  90];
        case 2,  v = [0,     0, 50,  0,    0,    0,  3, 45,  90];
        case 3,  v = [0,     0,  0,  0,    0,  100,  3, 45,  90];
        case 4,  v = [0,   100,  0,  0,    0,  100,  3, 45,  90];
        case 5,  v = [0,     0, 50,  0,   50,    0,  3, 45,  90];
        case 6,  v = [1,   500,  0,  0,    0,    0,  5, 45,  90];
        case 7
            phi0  = 90;  beta0 = 45.3;  dl = 500;
            dix_v = dl * cosd(phi0) / tand(beta0);
            diy_v = dl * sind(phi0) / tand(beta0);
            v = [0, dl, 0, 0, dix_v, diy_v, 3, beta0, phi0];
        otherwise, return;
    end

    for k = 1:9
        lo = state.sl_h{k}.Min;  hi = state.sl_h{k}.Max;
        state.sl_h{k}.Value = max(lo, min(hi, v(k)));
    end
    fig.UserData = state;
end

% ===========================================================================
%%  HELPER: update_metrics_panel
% ===========================================================================
function update_metrics_panel(txt, roe0, metrics, chief, show_uv, phi_0, beta)
%UPDATE_METRICS_PANEL  Write formation quality metrics to the GUI text box.
%
%  Displayed quantities:
%    - Orbit type and period
%    - R-N minimum and maximum separation (passive safety indicator)
%    - |de|, |di|, angle between eccentricity and inclination vectors
%    - Safety flag (passively safe / drifting / no e/i separation)
%    - Bmax/Bmin ratio and peak OPD (when UV panel is enabled)

    gd  = char(948);   % δ
    deg = char(176);   % °
    sep = repmat(char(8212), 1, 21);   % ─────── separator line

    n    = sqrt(chief.mu / chief.a^3);
    T_hr = (2*pi / n) / 3600;
    if chief.a > 30000e3
        orbit_lbl = 'GEO';
    else
        orbit_lbl = 'LEO';
    end

    % Safety flag and background colour
    if metrics.passive_safe
        safe_str = [char(10003) ' PASSIVELY SAFE'];
        txt.BackgroundColor = [0.84, 0.95, 0.84];
    elseif abs(roe0.da) > 0.5
        safe_str = [char(10007) ' DRIFTING  (' gd 'a' char(8800) '0)'];
        txt.BackgroundColor = [0.99, 0.87, 0.83];
    elseif metrics.de_norm < 0.5 && metrics.di_norm < 0.5
        safe_str = [char(9888) '  No e/i separation'];
        txt.BackgroundColor = [0.99, 0.96, 0.80];
    else
        safe_str = [char(10007) ' Not passive safe'];
        txt.BackgroundColor = [0.99, 0.87, 0.83];
    end

    lines = {
        sprintf('%s  |  T = %.2f hr', orbit_lbl, T_hr), ...
        sprintf('n = %.4e rad/s', n), ...
        sep, ...
        sprintf('RN min  = %.1f m', metrics.rn_min_sep), ...
        sprintf('RN max = %.1f m',  metrics.rn_max_sep), ...
        sprintf('|%se| = %.1f m',   gd, metrics.de_norm), ...
        sprintf('|%si|  = %.1f m',  gd, metrics.di_norm), ...
    };

    if ~isnan(metrics.angle_de_di)
        lines{end+1} = sprintf('%s(e,i) = %.1f%s', char(8736), ...
                                metrics.angle_de_di, deg);
    end
    lines{end+1} = sep;
    lines{end+1} = safe_str;

    if show_uv && ~isnan(metrics.Bmax_Bmin)
        lines{end+1} = sep;
        lines{end+1} = sprintf('B_max/B_min = %.2f', metrics.Bmax_Bmin);
        lines{end+1} = sprintf('Peak OPD = %.1f m',  metrics.peak_opd);
        lines{end+1} = sprintf('%s0=%d%s  %s=%d%s', ...
            char(966), round(phi_0), deg, char(946), round(beta), deg);
    end

    txt.String = strjoin(lines, newline);
end
