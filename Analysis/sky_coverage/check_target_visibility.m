function is_visible = check_target_visibility( ...
        target_unit, ~, ...
        sun_unit, moon_unit, earth_unit, ...
        earth_ang_radius, in_eclipse, params)
% target_unit can be Nx3 or 3x1

    if size(target_unit,2) ~= 3
        if isequal(size(target_unit),[3,1])
            target_unit = target_unit.';
        else
            error('target_unit must be Nx3 or 3x1');
        end
    end

    N = size(target_unit,1);

    if in_eclipse
        is_visible = false(N,1);
        return
    end

    earth_keepout = earth_ang_radius + params.exclusion.earth_margin;

    % dot-products (vectorized)
    dE = target_unit * earth_unit(:);
    dS = target_unit * sun_unit(:);
    dM = target_unit * moon_unit(:);

    % angle > threshold  <=> dot < cos(threshold)
    is_visible = ...
        (dE < cosd(earth_keepout)) & ...
        (dS < cosd(params.exclusion.sun)) & ...
        (dM < cosd(params.exclusion.moon));
end