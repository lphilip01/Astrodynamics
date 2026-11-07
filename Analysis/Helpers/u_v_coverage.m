% OCP solution: science interval only
idx = sol.scienceStartNode:numel(sol.t);
uvr = compute_uv_inertial(sol.chief, sol.RTN, ra, dec, idx);

figure; hold on; axis equal; grid on
for p = 1:size(uvr.pairs,1)
    plot( uvr.uv_m(:,1,p),  uvr.uv_m(:,2,p), '-',  'LineWidth', 1.5);
    plot(-uvr.uv_m(:,1,p), -uvr.uv_m(:,2,p), '--', 'LineWidth', 1.0, 'HandleVisibility','off');
end
xlabel('u [m]'); ylabel('v [m]');
fprintf('Projected baseline range [%.1f, %.1f] m, hull area %.3g m^2\n', ...
    uvr.projLenMin_m, uvr.projLenMax_m, uvr.hullArea_m2);

% Uncontrolled propagation:
% uvr0 = compute_uv_inertial(out.states.chief, out.dr_rtn, out.star.ra, out.star.dec);

outUV = out;
outUV.opd = num2cell(out.OPDfull_km, 1);   % 1-by-Nc cell of Nt-by-1 [km]
report = assess_uv_plane_coverage(outUV, sol);