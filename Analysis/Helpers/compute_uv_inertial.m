function uvr = compute_uv_inertial(chiefQNS, rtn_km, ra, dec, idx)
% compute_uv_inertial
% u-v tracks of collector-to-collector baselines in an inertial, sky-fixed frame.
%   chiefQNS : Nt-by-6 chief states [a ex ey inc RAAN u]  (out.states.chief or sol.chief)
%   rtn_km   : 1-by-Nc cell, each Nt-by-3 collector position in chief RTN [km]
%              (out.dr_rtn or sol.RTN)
%   ra, dec  : target right ascension, declination [rad]
%   idx      : optional sample indices (e.g. sol.scienceStartNode:numel(sol.t))
Nt = size(chiefQNS,1);
if nargin < 5 || isempty(idx), idx = (1:Nt).'; end
idx = idx(:);
Nc  = numel(rtn_km);
Ns  = numel(idx);

sI   = [cos(dec)*cos(ra); cos(dec)*sin(ra); sin(dec)];
uHat = cross([0;0;1], sI);  uHat = uHat/norm(uHat);   % east
vHat = cross(sI, uHat);                               % north

rI = repmat({zeros(Ns,3)}, 1, Nc);                    % inertial positions [m]
for n = 1:Ns
    k  = idx(n);
    inc = chiefQNS(k,4); O = chiefQNS(k,5); u = chiefQNS(k,6);
    cO = cos(O); sO = sin(O); ci = cos(inc); si = sin(inc);
    cu = cos(u); su = sin(u);
    Q = [cO*cu - sO*su*ci, -cO*su - sO*cu*ci,  sO*si; ...   % columns R, T, N
         sO*cu + cO*su*ci, -sO*su + cO*cu*ci, -cO*si; ...
         su*si,             cu*si,              ci];
    for j = 1:Nc
        rI{j}(n,:) = (1000*Q*rtn_km{j}(k,:).').';
    end
end

pairs = nchoosek(1:Nc, 2);
Np    = size(pairs,1);
uv    = zeros(Ns,2,Np);
w     = zeros(Ns,Np);
len3  = zeros(Ns,Np);
for p = 1:Np
    b = rI{pairs(p,2)} - rI{pairs(p,1)};
    uv(:,1,p) = b*uHat;
    uv(:,2,p) = b*vHat;
    w(:,p)    = b*sI;                  % component along the line of sight [m]
    len3(:,p) = sqrt(sum(b.^2,2));
end

allUV = reshape(permute(uv,[1 3 2]), [], 2);
pts   = unique([allUV; -allUV], 'rows');
area  = 0;
if size(pts,1) >= 3
    kh   = convhull(pts(:,1), pts(:,2));
    area = polyarea(pts(kh,1), pts(kh,2));
end
projLen = sqrt(squeeze(sum(uv.^2, 2)));

uvr.uv_m          = uv;                 % Ns x 2 x Np
uvr.pairs         = pairs;
uvr.uHat          = uHat;
uvr.vHat          = vHat;
uvr.sI            = sI;
uvr.w_m           = w;
uvr.len3D_m       = len3;
uvr.projLenMin_m  = min(projLen(:));
uvr.projLenMax_m  = max(projLen(:));
uvr.hullArea_m2   = area;               % includes conjugate points
uvr.nSamples      = Ns;
end