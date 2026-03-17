function metrics = compute_metrics(rtn, uv, opd, roe_traj)
%COMPUTE_METRICS  Formation flying safety and interferometric quality metrics.
%
% Computes four key metrics characterising the formation:
%
%  1. MINIMUM R-N SEPARATION (passive safety)
%     In the plane perpendicular to the along-track direction (R-N plane),
%     the trajectory must not pass through the origin (chief location) to
%     guarantee collision avoidance. The minimum R-N distance is the safety
%     margin under purely Keplerian dynamics.
%     Safe formations: min_rn_sep > 0  (i.e., some nonzero eccentricity or
%     inclination keeps the relative orbit away from the chief).
%
%  2. BASELINE RATIO Bmax/Bmin (interferometric quality)
%     Ratio of max to min perpendicular baseline magnitude over one or more
%     orbits. Unity = circular UV coverage (ideal). Large ratio = elongated
%     UV ellipse = anisotropic angular resolution.
%
%  3. PASSIVE SAFETY FLAG
%     True if: da=0 (no drift) AND (|de|>0 OR |di|>0).
%     This is the D'Amico e/i separation criterion: a drift-free formation
%     with nonzero relative eccentricity or inclination vectors is
%     passively safe (bounded and collision-avoiding under Keplerian dynamics).
%
%  4. PEAK / RMS OPD (interferometric phase stability)
%     Peak optical path difference over the orbit. For stellar interferometry,
%     OPD must be within the coherence length (~microns to mm). Large OPD
%     requires active fringe tracking or OPD compensation.
%
% Inputs:
%   rtn       - RTN relative position [m], 3 x N
%   uv        - UV plane coordinates [m], 2 x N  (empty [] if UV disabled)
%   opd       - Optical path difference [m], 1 x N  (empty [] if UV disabled)
%   roe_traj  - ROE time history [m], 6 x N
%
% Outputs:
%   metrics - Struct with fields (see below)

    % --- R-N plane separation ---
    rR = rtn(1, :);
    rN = rtn(3, :);
    rn_sep = sqrt(rR.^2 + rN.^2);    % R-N distance from chief [m]

    metrics.rn_min_sep     = min(rn_sep);     % minimum [m]
    metrics.rn_max_sep     = max(rn_sep);     % maximum [m]
    metrics.rn_separations = rn_sep;          % full time series [m]

    % --- UV / interferometric metrics ---
    if ~isempty(uv) && size(uv, 2) > 1
        B_mag = sqrt(uv(1,:).^2 + uv(2,:).^2);    % |B_perp| [m]
        metrics.B_perp_max = max(B_mag);
        metrics.B_perp_min = min(B_mag);
        if metrics.B_perp_min > 1e-6
            metrics.Bmax_Bmin = metrics.B_perp_max / metrics.B_perp_min;
        else
            metrics.Bmax_Bmin = Inf;
        end
    else
        metrics.B_perp_max = NaN;
        metrics.B_perp_min = NaN;
        metrics.Bmax_Bmin  = NaN;
    end

    % --- OPD metrics ---
    if ~isempty(opd)
        metrics.peak_opd = max(abs(opd));
        metrics.rms_opd  = sqrt(mean(opd.^2));
    else
        metrics.peak_opd = NaN;
        metrics.rms_opd  = NaN;
    end

    % --- ROE vector quantities ---
    dex = roe_traj(3, 1);
    dey = roe_traj(4, 1);
    dix = roe_traj(5, 1);
    diy = roe_traj(6, 1);
    da_mean = mean(roe_traj(1, :));

    metrics.de_norm = sqrt(dex^2 + dey^2);    % |relative eccentricity vector| [m]
    metrics.di_norm = sqrt(dix^2 + diy^2);    % |relative inclination vector| [m]

    % Angle between eccentricity and inclination vectors (e/i separation angle)
    % 90 deg = maximum R-N clearance; 0 or 180 deg = collinear (reduced clearance)
    if metrics.de_norm > 1e-4 && metrics.di_norm > 1e-4
        de_hat = [dex; dey] / metrics.de_norm;
        di_hat = [dix; diy] / metrics.di_norm;
        cos_ang = max(-1, min(1, dot(de_hat, di_hat)));
        metrics.angle_de_di = acosd(cos_ang);    % [deg]
    else
        metrics.angle_de_di = NaN;
    end

    % --- Passive safety flag ---
    % Condition 1: no secular drift (da ~ 0)
    da_zero  = abs(da_mean) < 0.5;                         % within 0.5 m
    % Condition 2: some nonzero eccentricity or inclination vector
    has_de   = metrics.de_norm > 0.5;                      % [m]
    has_di   = metrics.di_norm > 0.5;                      % [m]
    metrics.passive_safe = da_zero && (has_de || has_di);

    % --- Total baseline at each time step ---
    rT = rtn(2, :);
    metrics.total_sep_rms = sqrt(mean(rR.^2 + rT.^2 + rN.^2));
    metrics.total_sep_max = max(sqrt(rR.^2 + rT.^2 + rN.^2));
end
