%% Optional: Inclination Trade Study
inclinations = [0, 2, 5, 10, 15, 20];  % degrees
for idx = 1:length(inclinations)
    params.chief.ix = deg2rad(inclinations(idx));
    params.chief.iy = 0;
    % Run main analysis loop
    % Store metrics(idx) = ...
end

% Plot coverage vs inclination
figure;
plot(inclinations, [metrics.total_coverage_percent], '-o', 'LineWidth', 2);
xlabel('Inclination (deg)');
ylabel('Total Sky Coverage (%)');
title('Sky Coverage vs GEO Inclination');
grid on;