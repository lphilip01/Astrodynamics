function report = differential_perturbation_report(out,paramsChief,paramsDeputyCell,makePlot,outputBase)
% Differential perturbing acceleration in a COMMON inertial frame.
% Call after propagating the uncontrolled chief and collectors. Each
% breakdown supplies RTN acceleration [km/s^2] in its OWN orbital frame.
% The norm of (a_deputy_ECI - a_chief_ECI) is reported in nm/s^2.
% No thrust or central two-body gravity is included.

if nargin<4, makePlot = true; end
if nargin<5, outputBase = 'differential_perturbations_peak'; end
if ~iscell(paramsDeputyCell), paramsDeputyCell = {paramsDeputyCell}; end
t = out.t(:);
xc = out.states.chief;
xd = out.states.deputies;
if ~iscell(xd), xd = {xd}; end
Nt = numel(t); Nc = numel(xd);
if Nt < 2 || size(xc,1)~=Nt || size(xc,2)~=6 || ...
        numel(paramsDeputyCell)~=Nc || any(diff(t)<=0)
    error('Expected increasing out.t, Nt-by-6 states, and one parameter set per collector.');
end
for j=1:Nc
    if size(xd{j},1)~=Nt || size(xd{j},2)~=6
        error('Each collector state history must be Nt-by-6.');
    end
end

names = {'J2','SRP','Moon','Sun'};
nSrc = numel(names);
history = zeros(Nt,Nc,nSrc); % differential acceleration norms, nm/s^2
chiefParams = paramsChief;
deputyParams = paramsDeputyCell;

% Direct vectors avoid interpolating the same precomputed ephemeris once
% for each vehicle, source and output time.
directEphem = false;
if isfield(paramsChief,'ephem')
    ephem = paramsChief.ephem;
    if isfield(ephem,'t') && isfield(ephem,'rSun') && ...
            isfield(ephem,'rMoon')
        directEphem = numel(ephem.t)==Nt && ...
            all(abs(ephem.t(:)-t)<1e-6) && ...
            size(ephem.rSun,1)==Nt && size(ephem.rMoon,1)==Nt;
    end
end

for k=1:Nt
    if directEphem
        chiefParams.ephem.rSunNow  = ephem.rSun(k,:).';
        chiefParams.ephem.rMoonNow = ephem.rMoon(k,:).';
        for j=1:Nc
            deputyParams{j}.ephem.rSunNow  = ephem.rSun(k,:).';
            deputyParams{j}.ephem.rMoonNow = ephem.rMoon(k,:).';
        end
    end
    pc = qns_perturbation_breakdown(t(k),xc(k,:).',chiefParams);
    Qc = local_rtn_to_eci(xc(k,:).');
    for j=1:Nc
        pd = qns_perturbation_breakdown(t(k),xd{j}(k,:).',deputyParams{j});
        Qd = local_rtn_to_eci(xd{j}(k,:).');
        for p=1:nSrc
            name = names{p};
            da_eci = Qd*pd.accel.(name) - Qc*pc.accel.(name);
            history(k,j,p) = 1e12*norm(da_eci); % km/s^2 -> nm/s^2
        end
    end
end

spacecraft = strings(Nc+1,1);
spacecraft(1) = "Chief (combiner)";
As_m2 = zeros(Nc+1,1);
mass_kg = zeros(Nc+1,1);
As_m2(1)=paramsChief.As; mass_kg(1)=paramsChief.m;
for j=1:Nc
    spacecraft(j+1)="Collector " + string(j);
    As_m2(j+1)=paramsDeputyCell{j}.As;
    mass_kg(j+1)=paramsDeputyCell{j}.m;
end
if any(mass_kg<=0), error('All spacecraft masses must be positive.'); end
report.areaMass = table(spacecraft,As_m2,mass_kg,As_m2./mass_kg, ...
    'VariableNames',{'Spacecraft','Area_m2','Mass_kg','AreaToMass_m2kg'});

source = strings(Nc*nSrc,1);
collector = zeros(Nc*nSrc,1);
peak_nmps2 = zeros(Nc*nSrc,1);
peakTime_h = zeros(Nc*nSrc,1);
row=0;
for p=1:nSrc
    for j=1:Nc
        row=row+1;
        source(row)=string(names{p});
        collector(row)=j;
        [peak_nmps2(row),idx]=max(history(:,j,p));
        peakTime_h(row)=t(idx)/3600;
    end
end
report.peaks = table(source,collector,peak_nmps2,peakTime_h, ...
    'VariableNames',{'Source','Collector','PeakDifferential_nmps2','PeakTime_h'});
peakMatrix = reshape(peak_nmps2,Nc,nSrc).'; % source rows, collector columns
sourceMax = max(peakMatrix,[],2);
report.sourcePeaks = table(string(names(:)),sourceMax, ...
    'VariableNames',{'Source','PeakAcrossCollectors_nmps2'});
report.SRPisLargest = sourceMax(2)>max(sourceMax([1 3 4]));
report.SRPtoNextLargest = sourceMax(2)/max(sourceMax([1 3 4]));
report.figureCaption = sprintf([ ...
    'Peak differential acceleration over %.1f h of uncontrolled ', ...
    'formation propagation. Values are the maxima of the norm of ', ...
    'collector-minus-combiner perturbing acceleration, computed in ', ...
    'a common inertial frame for each source.'],(t(end)-t(1))/3600);
report.t_s=t;
report.sources=names;
report.differentialNorm_nmps2=history;

fprintf('\nSimulated area-to-mass ratios (chief and collectors):\n');
disp(report.areaMass);
fprintf('Peak |a_collector - a_chief| in a common ECI frame [nm/s^2]:\n');
disp(report.peaks);
fprintf('Peak per source across all collectors [nm/s^2]:\n');
disp(report.sourcePeaks);
if report.SRPisLargest
    fprintf('SRP peak is %.3g times the next-largest source peak.\n', ...
        report.SRPtoNextLargest);
else
    fprintf('SRP is not the largest peak differential acceleration.\n');
end

if makePlot
    if ~ischar(outputBase) && ~(isstring(outputBase) && isscalar(outputBase))
        error('outputBase must be a file path without an extension.');
    end
    outputBase = char(outputBase);
    % Match Example_GEO_Formation_OPD: 5-inch figure, Times New Roman,
    % colorblind-friendly palette, vector PDF and 600-dpi PNG exports.
    palette = [0.0000 0.4470 0.7410;
               0.8500 0.3250 0.0980;
               0.4660 0.6740 0.1880];
    if Nc>size(palette,1), palette=lines(Nc); end
    fig=figure('Name','Peak differential accelerations', ...
        'Units','inches','Position',[1 1 5 4.0],'Color','w');
    ax=axes(fig,'Position',[0.16 0.24 0.80 0.69]);

    % A zero differential has no position on a logarithmic axis. Keep
    % exact zeros in the numerical table and omit only their plotted bars.
    positive=peakMatrix(peakMatrix>0 & isfinite(peakMatrix));
    if isempty(positive)
        warning('All differential accelerations are zero; no log plot exported.');
        return
    end
    lowerLimit=10^floor(log10(min(positive)))-eps(min(positive));
    upperLimit=10^ceil(log10(max(positive)*1.2));
    if upperLimit<=lowerLimit, upperLimit=lowerLimit*10; end
    plotted=peakMatrix;
    plotted(plotted<=0)=NaN;
    h=bar(ax,plotted,'grouped','BarWidth',0.82);
    for j=1:Nc
        h(j).FaceColor=palette(j,:);
        h(j).EdgeColor='none';
        h(j).BaseValue=lowerLimit;
    end
    set(ax,'YScale','log','YLim',[lowerLimit upperLimit], ...
        'XLim',[0.5 nSrc+0.5],'XTick',1:nSrc, ...
        'XTickLabel',{'Earth $J_2$','SRP','Moon','Sun'}, ...
        'FontName','Times New Roman','FontSize',11, ...
        'TickLabelInterpreter','latex','LineWidth',0.8, ...
        'Box','on','TickDir','in','YGrid','on', ...
        'GridLineStyle','--','GridAlpha',0.25);
    ylabel(ax,'Peak differential acceleration [nm s$^{-2}$]', ...
        'Interpreter','latex');
    legend(ax,h,cellstr("Collector "+string(1:Nc)), ...
        'FontName','Times New Roman','FontSize',9, ...
        'Location','northwest','Box','off');

    ratios=As_m2./mass_kg;
    if all(abs(ratios(2:end)-ratios(2))<1e-12)
        deputyNote=sprintf('Collectors 1--%d: $A_s/m=%.4g$ m$^2$/kg', ...
            Nc,ratios(2));
    else
        deputyNote=['Collectors 1--' num2str(Nc) ': $A_s/m=$ ' ...
            char(strjoin(compose('%.4g',ratios(2:end).'),', ')) ' m$^2$/kg'];
    end
    note={sprintf('Chief: $A_s/m=%.4g$ m$^2$/kg',ratios(1)),deputyNote};
    annotation(fig,'textbox',[0.16 0.025 0.80 0.13], ...
        'String',note,'Interpreter','latex', ...
        'FontName','Times New Roman','FontSize',9, ...
        'EdgeColor','none','HorizontalAlignment','center');

    report.figurePDF=[outputBase '.pdf'];
    report.figurePNG=[outputBase '.png'];
    exportgraphics(fig,report.figurePDF, ...
        'ContentType','vector','BackgroundColor','white');
    exportgraphics(fig,report.figurePNG, ...
        'Resolution',600,'BackgroundColor','white');
end
end

function Q=local_rtn_to_eci(x)
% Exactly the right-handed near-circular basis used by the breakdown.
i=x(4); O=x(5); u=x(6);
cO=cos(O); sO=sin(O); ci=cos(i); si=sin(i); cu=cos(u); su=sin(u);
R=[cO*cu-sO*su*ci; sO*cu+cO*su*ci; su*si];
T=[-cO*su-sO*cu*ci; -sO*su+cO*cu*ci; cu*si];
N=[sO*si; -cO*si; ci];
Q=[R,T,N];
end
