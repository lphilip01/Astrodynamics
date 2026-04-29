mu = 398600;          % km^3/s^2
RE = 6378;            % km
J2 = 1082.63e-6;
deg = pi/180;

%% ------------------------------------------------------------------------
% Chief orbit (GEO, frozen)
%% ------------------------------------------------------------------------
a0    = 42164;               % km
e0    = 1e-4;
inc0  = 10 * deg;
RAAN0 = 0.0 * deg;
w0    = 0.0 * deg;
M0    = 0.0;
u0    = w0 + M0;

ex0 = e0*cos(w0);
ey0 = e0*sin(w0);
xc0 = [a0; ex0; ey0; inc0; RAAN0; u0];

%% ------------------------------------------------------------------------
% Sweep setup
%% ------------------------------------------------------------------------
ra = linspace(0, 2*pi, 5);
ra(end) = [];
dec = linspace(-pi/2, pi/2, 5);

rho_m = 1000;
rho = rho_m / 1000;
gammaVals = (2*pi/3) * (1:3);

Asd = 2;     % m^2
md  = 200;   % kg

scriptDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'results');
resultsFile = fullfile(resultsDir, 'star_direction_sweep_results.mat');

forceRecompute = false;  % Set true after changing sweep or dynamics inputs
savePlotFiles = true;

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

%% ------------------------------------------------------------------------
% Parameters
%% ------------------------------------------------------------------------
% These parameters can be set outside of the loop.
paramsChief.mu      = mu;
paramsChief.RE      = RE;
paramsChief.J2      = J2;
paramsChief.muMoon  = 4903;       % km^3/s^2
paramsChief.muSun   = 132712;     % km^3/s^2
paramsChief.CR      = 2;
paramsChief.As      = 2;          % m^2
paramsChief.m       = 200;        % kg
paramsChief.S       = 1367;       % W/m^2
paramsChief.c       = 2.998e8;    % m/s
paramsChief.jd0     = juliandate(datetime(2026,1,1,0,0,0));
paramsChief.ephemModel = '421';
paramsChief.useShadow = true;

paramsDeputy = paramsChief;
paramsDeputy.As = Asd;
paramsDeputy.m = md;

paramsDeputy2 = paramsChief;
paramsDeputy2.As = Asd;
paramsDeputy2.m = md;

paramsDeputy3 = paramsChief;
paramsDeputy3.As = Asd;
paramsDeputy3.m = md;

%% ------------------------------------------------------------------------
% Time span: 1 orbit
%% ------------------------------------------------------------------------
T0 = 2*pi*sqrt(a0^3/mu);
nOrbits = 1;
tf = nOrbits * T0;

nout  = 700;
tspan = linspace(0, tf, nout);

% Optional ephemeris precompute for compatibility with full pipeline
paramsChief.ephem   = precompute_ephemeris(tspan, paramsChief);
paramsDeputy.ephem  = paramsChief.ephem;
paramsDeputy2.ephem = paramsChief.ephem;
paramsDeputy3.ephem = paramsChief.ephem;

opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'InitialStep',T0/1000);

%% ------------------------------------------------------------------------
% Either load a cached sweep or propagate all RA / Dec combinations
%% ------------------------------------------------------------------------
if ~forceRecompute && isfile(resultsFile)
    load(resultsFile, 'sweepResults');
    fprintf('Loaded cached star-direction sweep from:\n  %s\n', resultsFile);
else
    fprintf('Running star-direction sweep for %d RA values x %d Dec values...\n', numel(ra), numel(dec));

    templateCase = struct( ...
        'ra', [], ...
        'dec', [], ...
        'phi0', [], ...
        'beta', [], ...
        'roe0', [], ...
        'roe02', [], ...
        'roe03', [], ...
        't', [], ...
        'xc', [], ...
        'xd', [], ...
        'xd2', [], ...
        'xd3', [], ...
        'roe_k', [], ...
        'roe_k2', [], ...
        'roe_k3', [], ...
        'dr_rtn1', [], ...
        'dr_rtn2', [], ...
        'dr_rtn3', []);

    sweepResults = struct();
    sweepResults.ra = ra;
    sweepResults.dec = dec;
    sweepResults.rho_m = rho_m;
    sweepResults.gammaVals = gammaVals;
    sweepResults.nOrbits = nOrbits;
    sweepResults.tspan = tspan;
    sweepResults.forceRecompute = forceRecompute;
    sweepResults.cases = repmat(templateCase, numel(ra), numel(dec));

    for ii = 1:numel(ra)
        for jj = 1:numel(dec)
            [phi0, beta, ~] = radec_to_phibeta(ra(ii), dec(jj), RAAN0, inc0);

            roeCell = cell(1, numel(gammaVals));
            for kk = 1:numel(gammaVals)
                gamma = gammaVals(kk);

                delta_ex = rho * (cos(gamma)*sin(phi0) + sin(gamma)*sin(beta)*cos(phi0));
                delta_ey = -(rho/2) * (cos(gamma)*cos(phi0) - sin(gamma)*sin(beta)*sin(phi0));
                delta_iy = -rho * sin(gamma) * cos(beta);

                roeCell{kk} = zeros(6,1);
                roeCell{kk}(1) = 0.0;
                roeCell{kk}(2) = 0.0;
                roeCell{kk}(3) = delta_ex / a0;
                roeCell{kk}(4) = delta_ey / a0;
                roeCell{kk}(5) = 0.0;
                roeCell{kk}(6) = delta_iy / a0;
            end

            xd0 = deputy_cell_from_chief_roe_qns(xc0, roeCell);

            [t, xc] = ode45(@(t,x) rates_qns_total(t,x,paramsChief),  tspan, xc0, opts);
            [~, xd] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy),  tspan, xd0{1}, opts);
            [~, xd2] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy2), tspan, xd0{2}, opts);
            [~, xd3] = ode45(@(t,x) rates_qns_total(t,x,paramsDeputy3), tspan, xd0{3}, opts);

            roe_k  = roe_from_qns_chief_deputy(xc, xd);
            roe_k2 = roe_from_qns_chief_deputy(xc, xd2);
            roe_k3 = roe_from_qns_chief_deputy(xc, xd3);

            dr_rtn  = rtn_from_roe(roe_k,  xc(:,1), xc(:,6));
            dr_rtn2 = rtn_from_roe(roe_k2, xc(:,1), xc(:,6));
            dr_rtn3 = rtn_from_roe(roe_k3, xc(:,1), xc(:,6));

            sweepResults.cases(ii,jj).ra = ra(ii);
            sweepResults.cases(ii,jj).dec = dec(jj);
            sweepResults.cases(ii,jj).roe0 = roeCell{1};
            sweepResults.cases(ii,jj).roe02 = roeCell{2};
            sweepResults.cases(ii,jj).roe03 = roeCell{3};
            sweepResults.cases(ii,jj).t = t;
            sweepResults.cases(ii,jj).xc = xc;
            sweepResults.cases(ii,jj).xd = xd;
            sweepResults.cases(ii,jj).xd2 = xd2;
            sweepResults.cases(ii,jj).xd3 = xd3;
            sweepResults.cases(ii,jj).roe_k = roe_k;
            sweepResults.cases(ii,jj).roe_k2 = roe_k2;
            sweepResults.cases(ii,jj).roe_k3 = roe_k3;
            sweepResults.cases(ii,jj).dr_rtn1 = dr_rtn;
            sweepResults.cases(ii,jj).dr_rtn2 = dr_rtn2;
            sweepResults.cases(ii,jj).dr_rtn3 = dr_rtn3;

            fprintf('  Completed RA %5.1f deg, Dec %+5.1f deg\n', rad2deg(ra(ii)), rad2deg(dec(jj)));
        end
    end

    save(resultsFile, 'sweepResults', '-v7.3');
    fprintf('Saved star-direction sweep to:\n  %s\n', resultsFile);
end

%% ------------------------------------------------------------------------
% Plot the varying RA values with the 5 different Dec values in each plot
%% ------------------------------------------------------------------------
colors = lines(numel(sweepResults.dec));

for ii = 1:numel(sweepResults.ra)
    fig = figure('Color', 'w', 'Name', sprintf('RA_%02d', ii));
    ax = axes(fig);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');

    for jj = 1:numel(sweepResults.dec)
        caseData = sweepResults.cases(ii,jj);
        plotColor = colors(jj,:);

        plot3(ax, caseData.dr_rtn1(:,1), caseData.dr_rtn1(:,2), caseData.dr_rtn1(:,3), ...
            '-', 'Color', plotColor, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Dec = %+5.1f deg', rad2deg(caseData.dec)));

        plot3(ax, caseData.dr_rtn2(:,1), caseData.dr_rtn2(:,2), caseData.dr_rtn2(:,3), ...
            '-', 'Color', plotColor, 'LineWidth', 1.2, 'HandleVisibility', 'off');

        plot3(ax, caseData.dr_rtn3(:,1), caseData.dr_rtn3(:,2), caseData.dr_rtn3(:,3), ...
            '-', 'Color', plotColor, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end

    xlabel(ax, 'R [km]');
    ylabel(ax, 'T [km]');
    zlabel(ax, 'N [km]');
    title(ax, { ...
        sprintf('RTN Formation Trajectories, RA = %5.1f deg', rad2deg(sweepResults.ra(ii)))});
    legend(ax, 'Location', 'bestoutside');
    view(ax, 3);

    if savePlotFiles
        figFile = fullfile(resultsDir, sprintf('star_direction_ra_%02d.fig', ii));
        pngFile = fullfile(resultsDir, sprintf('star_direction_ra_%02d.png', ii));
        savefig(fig, figFile);
        exportgraphics(ax, pngFile, 'Resolution', 300);
    end
end
