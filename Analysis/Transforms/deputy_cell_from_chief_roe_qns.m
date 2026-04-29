function xdCell = deputy_cell_from_chief_roe_qns(xc0, roeIn)
% deputy_cell_from_chief_roe_qns
%
% Generate a cell array of deputy QNS initial states from one chief QNS
% initial state and multiple ROE initial conditions.
%
% Inputs:
%   xc0   : 6x1 or 1x6 chief QNS initial state
%           xc0 = [a, ex, ey, i, RAAN, u]
%
%   roeIn : either
%           (1) cell array, each cell 6x1 or 1x6 ROE vector
%           (2) numeric matrix N x 6, each row one ROE vector
%
%           ROE ordering:
%             [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]
%
% Output:
%   xdCell : 1xN cell array of deputy QNS initial states, each 6x1
%
% Notes:
%   Uses near-circular ROE-to-deputy initialization:
%       delta_a   = (ad-ac)/ac
%       delta_lambda = (ud-uc) + (RAANd-RAANc) cos(ic)
%       delta_ex  = exd - exc
%       delta_ey  = eyd - eyc
%       delta_ix  = id - ic
%       delta_iy  = (RAANd-RAANc) sin(ic)
%

xc0 = xc0(:);   % force column

if numel(xc0) ~= 6
    error('xc0 must be a 6-element QNS state.');
end

% Convert numeric matrix input to cell array
if isnumeric(roeIn)
    if size(roeIn,2) ~= 6
        error('If roeIn is numeric, it must be N x 6.');
    end
    nDep = size(roeIn,1);
    roeCell = cell(1,nDep);
    for k = 1:nDep
        roeCell{k} = roeIn(k,:).';
    end
elseif iscell(roeIn)
    nDep = numel(roeIn);
    roeCell = cell(1,nDep);
    for k = 1:nDep
        roeCell{k} = roeIn{k}(:);
        if numel(roeCell{k}) ~= 6
            error('Each ROE cell must contain a 6-element vector.');
        end
    end
else
    error('roeIn must be either a cell array or an N x 6 numeric matrix.');
end

% Build deputy cell array
xdCell = cell(1,nDep);
for k = 1:nDep
    xdCell{k} = local_deputy_from_chief_roe_qns(xc0, roeCell{k});
end

end

% =========================================================================
function xd0 = local_deputy_from_chief_roe_qns(xc0, roe0)
% Build one deputy QNS state from one chief QNS state and one ROE vector

ac  = xc0(1);
exc = xc0(2);
eyc = xc0(3);
ic  = xc0(4);
RAc = xc0(5);
uc  = xc0(6);

da   = roe0(1);
dl   = roe0(2);
dex  = roe0(3);
dey  = roe0(4);
dix  = roe0(5);
diy  = roe0(6);

ad  = ac * (1 + da);
exd = exc + dex;
eyd = eyc + dey;
id  = ic + dix;

sin_ic = sin(ic);
if abs(sin_ic) < 1e-12
    sin_ic = sign(sin_ic + eps) * 1e-12;
end

dRA   = diy / sin_ic;
RAANd = RAc + dRA;
ud    = uc + dl - dRA*cos(ic);

xd0 = [ad; exd; eyd; id; RAANd; ud];
end