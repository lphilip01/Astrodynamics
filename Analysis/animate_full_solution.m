out2=out;
phi0=out2.star.phi0;
beta=out2.star.beta;

[kStart, info] = find_science_hold_start(out2, 5, 'bestLocal');


%need to inject solution values from kStart onwards, out.dr_rtn is used

%interp to correct time scale
sol_t=sol.t;
new_sol_t = t(kStart):10:(t(kStart)+sol_t(end));

dr_rtn_new={};
for ii=1:numel(sol.RTN)
    this_rtn=sol.RTN{ii};
    
    R = this_rtn(:,1);
    T=this_rtn(:,2);
    N=this_rtn(:,3);

    new_R = interp1(sol_t,R,0:10:sol_t(end));
    new_T = interp1(sol_t,T,0:10:sol_t(end));
    new_N = interp1(sol_t,N,0:10:sol_t(end));

    new_RTN = [new_R' new_T' new_N'];
    out2.dr_rtn{ii}=[out.dr_rtn{ii}(1:kStart,:);new_RTN];



end

sol_u=sol.chief(:,6);
new_sol_u = interp1(sol_t,sol_u,0:10:sol_t(end));



chief.u = [out2.states.chief(1:kStart,6); new_sol_u'];

t = [out2.t(1:kStart); new_sol_t'];




opts = struct();
opts.step = 10;
opts.showPlane = true;
opts.pauseTime = 0.02;
opts.trailLength = 30;

animate_formation_star_plane(t, out2, chief, [phi0 beta], 'phibeta', opts);