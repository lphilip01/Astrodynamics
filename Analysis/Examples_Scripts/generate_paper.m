% Load the six saved runs
names  = {'tau_ceti','regulus','eltanin','alphecca'};
outs = {}; sols = {}; labels = {};
for n = 1:numel(names)
    for T = [7200]
        a = load(sprintf('out_%d_int_3600_settle_%s.mat', T, names{n}));
        b = load(sprintf('sol_%d_int_3600_settle_%s.mat', T, names{n}));
        outs{end+1} = a.out;  sols{end+1} = b.sol;
    end
end
R = generate_paper_results(outs, sols, ...
        'Tint_s', repmat([7200], 1, 3), ...      % needed only if sol lacks scienceStartNode/info
        'SkyFile', 'sky_coverage_results_24_0.mat');