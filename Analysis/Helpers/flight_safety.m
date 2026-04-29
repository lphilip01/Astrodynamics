function [stats, ps, margin] = flight_safety(out)
rtn_cell = out.dr_rtn;
for jj=1:numel(rtn_cell)
rtn_cell{jj} = rtn_cell{jj}.*10^3;
end
roe_cell = out.roe;
t_vec=out.t;

stats = pairwise_separation(rtn_cell);

% 1. Define KOZ (tune for your mission — these are illustrative GEO values)
koz_radii = [50, 150, 50];   % [R, T, N] meters — tighter radially

% 2. Check passive safety from ROE
ps = passive_safety_check(roe_cell, 0.5e-5,t_vec);   % min |δe| threshold

% 3. KOZ hard violation check
[viol, t_viol] = koz_check(rtn_cell, koz_radii, t_vec);

% 4. Margin timeline (soft check — trend toward violation)
margin = safety_margin_timeline(rtn_cell, koz_radii);

% 5. Find time to margin < threshold (e.g. 1.5x KOZ)
warning_threshold = 1.5;
t_warning = t_vec(find(margin < warning_threshold, 1));

% 6. Summary plot
plot_safety_summary(rtn_cell, roe_cell, t_vec, koz_radii);


function stats = pairwise_separation(rtn_cell)
    n = numel(rtn_cell);
    % Include chief as zero trajectory
    all_rtns = [{zeros(size(rtn_cell{1}, 1),3)}, rtn_cell(:)'];
    
    for i = 1:numel(all_rtns)
        for j = i+1:numel(all_rtns)
            dr = all_rtns{i} - all_rtns{j};          % 3xM
            sep = vecnorm(dr, 2, 1);                  % 1xM
            stats(i,j).min_sep    = min(sep);
            stats(i,j).min_idx    = find(sep == min(sep), 1);
            stats(i,j).mean_sep   = mean(sep);
            stats(i,j).trajectory = sep;
        end
    end
end

function [violated, t_viol] = koz_check(rtn_cell, koz_radii, t_vec)
    % koz_radii: [R_r, R_t, R_n] semi-axes in meters
    % Returns logical matrix and violation timestamps
    
    n = numel(rtn_cell);
    violated = false(n, n);
    t_viol   = cell(n, n);
    
    all_rtns = [{zeros(size(rtn_cell{1}, 1),3)}, rtn_cell(:)'];
    
    S = diag(1 ./ koz_radii.^2);   % ellipsoid metric
    
    for i = 1:numel(all_rtns)
        for j = i+1:numel(all_rtns)
            dr = all_rtns{i} - all_rtns{j};   % 3xM
            % Mahalanobis-like ellipsoid test: sum((dr/a)^2) < 1
            inside = sum((S * dr') .* dr', 1) < 1;
            violated(i,j) = any(inside);
            t_viol{i,j}   = t_vec(inside);
        end
    end
end

function safe = passive_safety_check(roe_cell, min_ecc_sep,t_vec)
    % roe_cell elements: 6xM [da, dlambda, dex, dey, dix, diy]
    % min_ecc_sep: minimum |delta_e| * a threshold in meters
    
    n = numel(roe_cell);
    safe = struct();
    
    for k = 1:n
        roe = roe_cell{k};   % 6xM
        
        de = roe(:,3:4);                    % delta_e vector over time
        di = roe(:,5:6);                    % delta_i vector over time
        
        de_mag = vecnorm(de, 1, 2);          % 1xM
        di_mag = vecnorm(di, 1, 2);
        
        % Phase angle between de and di (key for 3D safety)
        phase = atan2d(de(:,2), de(:,1)) - atan2d(di(:,2), di(:,1));
        
        safe(k).de_mag        = de_mag;
        safe(k).di_mag        = di_mag;
        safe(k).min_de        = min(de_mag);
        safe(k).phase_deg     = phase;
        safe(k).is_passive    = all(de_mag > min_ecc_sep);
        safe(k).drift_rate    = gradient(de_mag) / gradient(t_vec);  % d|δe|/dt
    end
end

function margin = safety_margin_timeline(rtn_cell, koz_radii)
    all_rtns = [{zeros(size(rtn_cell{1}, 1),3)}, rtn_cell(:)'];
    S = diag(1 ./ koz_radii.^2);
    M_t = size(rtn_cell{1}, 1);
    
    % Worst-case (minimum) ellipsoidal distance across all pairs at each timestep
    margin = inf(1, M_t);
    
    for i = 1:numel(all_rtns)
        for j = i+1:numel(all_rtns)
            dr = all_rtns{i} - all_rtns{j};
            % Normalized ellipsoidal distance: 1.0 = KOZ boundary
            d_ellip = sqrt(sum((S * dr') .* dr', 1));
            margin = min(margin, d_ellip);
        end
    end
    % margin < 1 → KOZ violated; margin = 2 → 2x KOZ radius clearance
end

function plot_safety_summary(rtn_cell, roe_cell, t_vec, koz_radii)
    figure('Position', [100 100 1400 900]);
    
    % --- Panel 1: Pairwise separation over time ---
    subplot(2,3,1);
    all_rtns = [{zeros(size(rtn_cell{1}, 1),3)}, rtn_cell(:)'];
    names = ['Chief', arrayfun(@(k) sprintf('Dep %d',k), 1:numel(rtn_cell), 'uni',0)];
    hold on;
    for i = 1:numel(all_rtns)
        for j = i+1:numel(all_rtns)
            dr = all_rtns{i} - all_rtns{j};
            plot(t_vec/3600, vecnorm(dr,1,2), 'DisplayName', ...
                 sprintf('%s-%s', names{i}, names{j}));
        end
    end
    yline(min(koz_radii), 'r--', 'KOZ');
    xlabel('Time (hr)'); ylabel('Sep (m)'); title('Pairwise Separation');
    legend('Location','best'); grid on;
    
    % --- Panel 2: Safety margin (ellipsoidal) ---
    subplot(2,3,2);
    margin = safety_margin_timeline(rtn_cell, koz_radii);
    plot(t_vec/3600, margin, 'k', 'LineWidth', 1.5);
    yline(1, 'r--', 'KOZ boundary');
    yline(2, 'g--', '2x margin');
    xlabel('Time (hr)'); ylabel('Normalized margin'); title('Worst-Case Safety Margin');
    grid on;
    
    % --- Panel 3: delta_e magnitude per deputy ---
    subplot(2,3,3);
    hold on;
    for k = 1:numel(roe_cell)
        de_mag = vecnorm(roe_cell{k}(:,3:4), 1, 2);
        plot(t_vec/3600, de_mag, 'DisplayName', sprintf('Dep %d', k));
    end
    xlabel('Time (hr)'); ylabel('|\delta e|'); title('Rel. Eccentricity Magnitude');
    legend('Location','best'); grid on;
    
    % --- Panel 4: delta_e phase ---
    subplot(2,3,4);
    hold on;
    for k = 1:numel(roe_cell)
        roe = roe_cell{k};
        phase = atan2d(roe(:,4), roe(:,3));
        plot(t_vec/3600, unwrap(phase*pi/180)*180/pi, ...
             'DisplayName', sprintf('Dep %d', k));
    end
    xlabel('Time (hr)'); ylabel('\phi_e (deg)'); title('\delta e Phase (SRP drift)');
    legend('Location','best'); grid on;
    
    % --- Panel 5: ROE plane (de_x vs de_y) ---
    subplot(2,3,5);
    hold on;
    for k = 1:numel(roe_cell)
        roe = roe_cell{k};
        plot(roe(:,3), roe(:,4), 'DisplayName', sprintf('Dep %d', k));
        plot(roe(1,3), roe(1,4), 'ko', 'MarkerSize', 5);  % epoch
    end
    plot(0, 0, 'r+', 'MarkerSize', 10, 'LineWidth', 2);  % origin = unsafe
    xlabel('\delta e_x'); ylabel('\delta e_y'); title('\delta e Trajectory');
    axis equal; legend('Location','best'); grid on;
    
    % --- Panel 6: RTN 3D closest approach ---
    subplot(2,3,6);
    hold on;
    cmap = lines(numel(rtn_cell));
    for k = 1:numel(rtn_cell)
        r = rtn_cell{k};
        plot3(r(:,1), r(:,2), r(:,3), 'Color', cmap(k,:), ...
              'DisplayName', sprintf('Dep %d', k));
    end
    plot3(0,0,0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
    xlabel('R (m)'); ylabel('T (m)'); zlabel('N (m)');
    title('RTN Trajectories'); legend('Location','best'); grid on; view(30,20);
end

end