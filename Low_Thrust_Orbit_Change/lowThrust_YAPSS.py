"""
YAPSS solution of the low-thrust orbit transfer problem.

Transcribed from the GPOPS-II reference example (Betts 2009).
State: modified equinoctial elements + spacecraft mass (7 states)
Control: unit thrust direction [ur, ut, uh] (3 controls)
Parameter: throttle tau in [-50, 0]  (ns=1)
Path constraint: ||u||^2 = 1
Terminal constraints: p_f, eccentricity magnitude, inclination magnitude,
                      cross term (equality), sign inequality (nd=5)

Units: SI throughout (m, kg, s, N)
"""

from __future__ import annotations

__all__ = ["main", "plot_solution", "setup"]

# standard library imports
from math import pi

# third party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# package imports
from yapss import ContinuousArg, DiscreteArg, ObjectiveArg, Problem, Solution
from yapss.math import cos, sin, sqrt

# =============================================================================
# Unit conversions
# =============================================================================


# =============================================================================
# Physical constants (SI)
# =============================================================================
mu   = 3.9860e14   # 3.9860e14  m^3/s^2
T    = 0.01978;      # 0.01978    N
Isp  = 450.0                       # s
g0   = 9.80665                     # m/s^2
Re   = 6.3781e6          # 6.3781e6   m
J2   =  1082.639e-6
J3   =    -2.565e-6
J4   =    -1.608e-6

# Initial/terminal conditions
p0   = 6.6559e6     # 6.6559e6  m
f0   = 0.0
g_0  = 0.0
h0   = -0.25396764647494
k0   = 0.0
L0   = pi
w0   = 0.4536             # 0.4536    kg

pf_tgt  = 1.219e7   # 1.219e7   m
ecc_tgt = 0.73550320568829
inc_tgt = 0.61761258786099

tf_guess = 90000.0                  # s

# =============================================================================
# Scaling (all scaled quantities are O(1))
# =============================================================================
p_scale = p0          # ~6.66e6 m
w_scale = w0          # 0.4536 kg
t_scale = tf_guess    # 90000  s
L_scale = 9 * 2 * pi  # max true longitude


# =============================================================================
# MEE dynamics helper (pure Python / yapss.math, returns tuple of 7 rates)
# Works with both numpy scalars and yapss AD scalars.
# =============================================================================
def mee_dynamics(p, f, g, h, k, L, w, ur, ut, uh, tau):
    """Return (dp, df, dg, dh, dk, dL, dw) for MEE + J2/J3/J4 + EP thrust."""
    # --- Auxiliary quantities ---
    q      = 1 + f * cos(L) + g * sin(L)
    r      = p / q
    alpha2 = h**2 - k**2
    s2     = 1 + h**2 + k**2

    # --- ECI position ---
    rX = (r / s2) * (cos(L) + alpha2 * cos(L) + 2 * h * k * sin(L))
    rY = (r / s2) * (sin(L) - alpha2 * sin(L) + 2 * h * k * cos(L))
    rZ = (2 * r / s2) * (h * sin(L) - k * cos(L))

    # --- ECI velocity ---
    vX = -(1/s2)*sqrt(mu/p)*(sin(L)+alpha2*sin(L)-2*h*k*cos(L)+g-2*f*h*k+alpha2*g)
    vY = -(1/s2)*sqrt(mu/p)*(-cos(L)+alpha2*cos(L)+2*h*k*sin(L)-f+2*g*h*k+alpha2*f)
    vZ =  (2/s2)*sqrt(mu/p)*(h*cos(L)+k*sin(L)+f*h+g*k)

    rMag = sqrt(rX**2 + rY**2 + rZ**2)

    # --- RTN basis ---
    ir1, ir2, ir3 = rX/rMag, rY/rMag, rZ/rMag

    # h_vec = r x v
    hv1 = rY*vZ - rZ*vY
    hv2 = rZ*vX - rX*vZ
    hv3 = rX*vY - rY*vX
    hMag = sqrt(hv1**2 + hv2**2 + hv3**2)
    ih1, ih2, ih3 = hv1/hMag, hv2/hMag, hv3/hMag

    # it = ih x ir
    it1 = ih2*ir3 - ih3*ir2
    it2 = ih3*ir1 - ih1*ir3
    it3 = ih1*ir2 - ih2*ir1

    # --- Oblateness perturbations ---
    sinphi = rZ / rMag
    cosphi = sqrt(1 - sinphi**2)

    P2  = (3*sinphi**2 - 1) / 2
    P3  = (5*sinphi**3 - 3*sinphi) / 2
    P4  = (35*sinphi**4 - 30*sinphi**2 + 3) / 8
    dP2 = 3*sinphi
    dP3 = (15*sinphi**2 - 3) / 2
    dP4 = (140*sinphi**3 - 60*sinphi) / 8

    # North unit vector
    en_dot_ir = ir3
    in1_u = -en_dot_ir*ir1
    in2_u = -en_dot_ir*ir2
    in3_u =  1 - en_dot_ir*ir3
    in_mag = sqrt(in1_u**2 + in2_u**2 + in3_u**2)
    in1, in2, in3 = in1_u/in_mag, in2_u/in_mag, in3_u/in_mag

    sumn = (Re/r)**2*dP2*J2 + (Re/r)**3*dP3*J3 + (Re/r)**4*dP4*J4
    sumr = 3*(Re/r)**2*P2*J2 + 4*(Re/r)**3*P3*J3 + 5*(Re/r)**4*P4*J4

    dg_n = -(mu*cosphi/r**2)*sumn
    dg_r = -(mu/r**2)*sumr

    # δg in ECI
    dg1 = dg_n*in1 - dg_r*ir1
    dg2 = dg_n*in2 - dg_r*ir2
    dg3 = dg_n*in3 - dg_r*ir3

    # Project to RTN
    Dg1 = ir1*dg1 + ir2*dg2 + ir3*dg3
    Dg2 = it1*dg1 + it2*dg2 + it3*dg3
    Dg3 = ih1*dg1 + ih2*dg2 + ih3*dg3

    # --- Thrust acceleration (SI: no g0 in numerator) ---
    T_eff = T * (1 + 0.01 * tau)
    at    = T_eff / w

    D1 = Dg1 + at * ur
    D2 = Dg2 + at * ut
    D3 = Dg3 + at * uh

    # --- MEE EOMs ---
    sq = sqrt(p / mu)

    dp = (2*p/q) * sq * D2

    df = (  sq*sin(L)*D1
          + sq*(1/q)*((q+1)*cos(L) + f)*D2
          - sq*(g/q)*(h*sin(L) - k*cos(L))*D3 )

    dg_mee = ( -sq*cos(L)*D1
              + sq*(1/q)*((q+1)*sin(L) + g)*D2
              + sq*(f/q)*(h*sin(L) - k*cos(L))*D3 )

    dh = sq * (s2*cos(L)/(2*q)) * D3
    dk = sq * (s2*sin(L)/(2*q)) * D3

    dL = ( sq*(1/q)*(h*sin(L) - k*cos(L))*D3
          + sqrt(mu*p)*(q/p)**2 )

    dw = -(T_eff / (Isp * g0))

    return dp, df, dg_mee, dh, dk, dL, dw


# =============================================================================
# Initial guess: propagate with tau=-25, velocity-aligned thrust
# =============================================================================
def generate_initial_guess(N_pts: int = 201):
    """Propagate MEE dynamics to generate initial guess on N_pts time grid."""

    def vel_aligned_u(p, f, g, h, k, L):
        """Unit thrust vector aligned with inertial velocity, in RTN."""
        s2     = 1 + h**2 + k**2
        alpha2 = h**2 - k**2
        q_val  = 1 + f*np.cos(L) + g*np.sin(L)
        r_val  = p / q_val

        rX = (r_val/s2)*(np.cos(L)+alpha2*np.cos(L)+2*h*k*np.sin(L))
        rY = (r_val/s2)*(np.sin(L)-alpha2*np.sin(L)+2*h*k*np.cos(L))
        rZ = (2*r_val/s2)*(h*np.sin(L)-k*np.cos(L))
        r_ECI = np.array([rX, rY, rZ]); rMag = np.linalg.norm(r_ECI)

        vX = -(1/s2)*np.sqrt(mu/p)*(np.sin(L)+alpha2*np.sin(L)-2*h*k*np.cos(L)+g-2*f*h*k+alpha2*g)
        vY = -(1/s2)*np.sqrt(mu/p)*(-np.cos(L)+alpha2*np.cos(L)+2*h*k*np.sin(L)-f+2*g*h*k+alpha2*f)
        vZ =  (2/s2)*np.sqrt(mu/p)*(h*np.cos(L)+k*np.sin(L)+f*h+g*k)
        v_ECI = np.array([vX, vY, vZ])

        ir = r_ECI / rMag
        hv = np.cross(r_ECI, v_ECI); hv /= np.linalg.norm(hv)
        it = np.cross(hv, ir)

        v_hat = v_ECI / np.linalg.norm(v_ECI)
        u_rtn = np.array([np.dot(v_hat, ir), np.dot(v_hat, it), np.dot(v_hat, hv)])
        return u_rtn / np.linalg.norm(u_rtn)

    def ode_rhs(t, x):
        p_,f_,g_,h_,k_,L_,w_ = x
        u = vel_aligned_u(p_, f_, g_, h_, k_, L_)
        rates = mee_dynamics(p_,f_,g_,h_,k_,L_,w_, u[0],u[1],u[2], tau=-25.0)
        return list(rates)

    x0 = [p0, f0, g_0, h0, k0, L0, w0]
    sol = solve_ivp(ode_rhs, [0, tf_guess], x0,
                   method='RK45', rtol=1e-8, atol=1e-10,
                   max_step=300, dense_output=True)

    t_grid = np.linspace(0, tf_guess, N_pts)
    X_grid = sol.sol(t_grid)       # shape (7, N_pts)

    # Enforce mass positivity
    X_grid[6, :] = np.maximum(X_grid[6, :], 0.01 * w0)

    # Control: velocity-aligned at each point
    U_grid = np.zeros((N_pts, 3))
    for i in range(N_pts):
        p_,f_,g_,h_,k_,L_ = X_grid[:6, i]
        U_grid[i, :] = vel_aligned_u(p_, f_, g_, h_, k_, L_)

    return t_grid, X_grid.T, U_grid   # X_grid.T: (N_pts, 7)


# =============================================================================
# YAPSS problem setup
# =============================================================================
def setup() -> Problem:
    """Set up the low-thrust MEE orbit transfer optimal control problem."""

    ocp = Problem(
        name  = "Low-Thrust MEE Orbit Transfer (GPOPS-II benchmark)",
        nx    = [7],    # [p, f, g, h, k, L, w]
        nu    = [3],    # [ur, ut, uh]
        nh    = [1],    # ||u||^2 = 1
        ns    = 1,      # tau (throttle parameter)
        nd    = 5,      # terminal constraints
        nq    = [0],
    )

    # -------------------------------------------------------------------------
    # Callback: objective — maximize final mass
    # -------------------------------------------------------------------------
    def objective(arg: ObjectiveArg) -> None:
        arg.objective = -arg.phase[0].final_state[6]   # -w_f (maximize mass)

    # -------------------------------------------------------------------------
    # Callback: continuous dynamics + path constraint
    # -------------------------------------------------------------------------
    def continuous(arg: ContinuousArg) -> None:
        p, f, g, h, k, L, w = arg.phase[0].state
        ur, ut, uh           = arg.phase[0].control
        tau                  = arg.parameter[0]

        dp, df, dg_mee, dh, dk, dL, dw = mee_dynamics(
            p, f, g, h, k, L, w, ur, ut, uh, tau)

        arg.phase[0].dynamics[:] = (dp, df, dg_mee, dh, dk, dL, dw)

        # Path constraint: unit thrust vector
        arg.phase[0].path[0] = ur**2 + ut**2 + uh**2

    # -------------------------------------------------------------------------
    # Callback: discrete (terminal) constraints
    # -------------------------------------------------------------------------
    def discrete(arg: DiscreteArg) -> None:
        ff = arg.phase[0].final_state[1]
        gf = arg.phase[0].final_state[2]
        hf = arg.phase[0].final_state[3]
        kf = arg.phase[0].final_state[4]
        pf = arg.phase[0].final_state[0]

        arg.discrete[0] = pf - pf_tgt                           # p_f = pf_tgt
        arg.discrete[1] = ff**2 + gf**2 - ecc_tgt**2           # ecc mag = target
        arg.discrete[2] = hf**2 + kf**2 - inc_tgt**2           # inc mag = target
        arg.discrete[3] = ff*hf + gf*kf                         # cross = 0
        arg.discrete[4] = gf*hf - kf*ff                         # <= 0 (inequality)

    # register functions
    ocp.functions.objective  = objective
    ocp.functions.continuous = continuous
    ocp.functions.discrete   = discrete

    # use automatic differentiation (exact, second order)
    ocp.derivatives.method = "auto"
    ocp.derivatives.order  = "second"

    # -------------------------------------------------------------------------
    # Bounds
    # -------------------------------------------------------------------------
    b = ocp.bounds.phase[0]

    # time
    b.initial_time.lower = b.initial_time.upper = 0.0
    b.final_time.lower   = 50000.0
    b.final_time.upper   = 120000.0

    # initial state (pinned)
    b.initial_state.lower[:] = b.initial_state.upper[:] = [p0, f0, g_0, h0, k0, L0, w0]

    # state path bounds
    p_min = 20e6 * 0.3048;  p_max = 60e6 * 0.3048
    b.state.lower[:] = [p_min, -1, -1, -1, -1, L0,  0.005]
    b.state.upper[:] = [p_max,  1,  1,  1,  1,  9*2*pi, w0]

    # terminal state: p pinned, mass lower bound, L monotone handled by state bounds
    b.final_state.lower[6] = 0.005       # mass > 0 at end
    b.final_state.upper[6] = w0

    # control bounds
    b.control.lower[:] = [-1, -1, -1]
    b.control.upper[:] = [ 1,  1,  1]

    # path constraint: ||u||^2 == 1
    b.path.lower[0] = 1.0
    b.path.upper[0] = 1.0

    # parameter (throttle tau)
    ocp.bounds.parameter.lower[0] = -50.0
    ocp.bounds.parameter.upper[0] =   0.0

    # discrete constraints
    ocp.bounds.discrete.lower[:4] = 0.0    # equalities: [0, 0, 0, 0]
    ocp.bounds.discrete.upper[:4] = 0.0
    ocp.bounds.discrete.lower[4]  = -float("inf")  # inequality: <= 0
    ocp.bounds.discrete.upper[4]  = 0.0

    # -------------------------------------------------------------------------
    # Initial guess (from ODE propagation)
    # -------------------------------------------------------------------------
    print("Generating initial guess...")
    t_g, X_g, U_g = generate_initial_guess(N_pts=101)
    print(f"  Done. Final mass in guess: {X_g[-1, 6]:.4f} kg")

    ocp.guess.phase[0].time    = t_g
    ocp.guess.phase[0].state   = X_g.T    # shape (101, 7)
    ocp.guess.phase[0].control = U_g.T    # shape (101, 3)
    ocp.guess.parameter        = [-25.0]

    # -------------------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------------------
    s = ocp.scale.phase[0]
    s.state[:]    = [p_scale, 1, 1, 1, 1, L_scale, w_scale]
    s.dynamics[:] = [p_scale/t_scale, 1/t_scale, 1/t_scale,
                     1/t_scale, 1/t_scale, L_scale/t_scale, w_scale/t_scale]
    s.time        = t_scale
    s.path[0]     = 1.0
    ocp.scale.objective = w_scale

    # -------------------------------------------------------------------------
    # Mesh: use moderately dense mesh, concentrate nodes by using more intervals
    # -------------------------------------------------------------------------
    ocp.spectral_method = "lgl"

    # 40 intervals of 6 collocation points each = 240 total collocation points
    # Comparable to N=200 multiple-shooting intervals in CasADi
    M, nc = 40, 6
    ocp.mesh.phase[0].collocation_points = M * (nc,)
    ocp.mesh.phase[0].fraction           = M * (1.0/M,)

    # -------------------------------------------------------------------------
    # IPOPT options
    # -------------------------------------------------------------------------
    ocp.ipopt_options.max_iter           = 2000
    ocp.ipopt_options.tol                = 1e-8
    ocp.ipopt_options.print_level        = 5
    ocp.ipopt_options.linear_solver      = "mumps"

    return ocp


# =============================================================================
# Plotting
# =============================================================================
def plot_solution(solution: Solution) -> None:
    """Plot the low-thrust orbit transfer solution."""
    t   = solution.phase[0].time
    tc  = solution.phase[0].time_c
    x   = solution.phase[0].state        # (7, N_state)
    u   = solution.phase[0].control      # (3, N_ctrl)
    tau_opt = solution.parameter[0]
    t_hr = t / 3600
    tc_hr = tc / 3600

    print("\n=== Solution Summary ===")
    print(f"  Objective (max mass): {-solution.objective:.6f} kg")
    print(f"  tau (throttle):       {tau_opt:.4f}")
    print(f"  Final time:           {t[-1]:.1f} s  ({t[-1]/3600:.3f} hr)")
    print(f"  Initial mass:         {x[6, 0]:.6f} kg")
    print(f"  Final mass:           {x[6, -1]:.6f} kg")
    print(f"  Fuel used:            {x[6,0]-x[6,-1]:.6f} kg")

    # Terminal constraint residuals
    ff, gf, hf, kf = x[1,-1], x[2,-1], x[3,-1], x[4,-1]
    print("\nTerminal constraint residuals:")
    print(f"  p error:   {x[0,-1] - pf_tgt:+.4e} m")
    print(f"  ecc error: {(ff**2+gf**2)**0.5 - ecc_tgt:+.4e}")
    print(f"  inc error: {(hf**2+kf**2)**0.5 - inc_tgt:+.4e}")
    print(f"  cross:     {ff*hf+gf*kf:+.4e}")
    print(f"  ineq:      {gf*hf-kf*ff:+.4e}")

    # --- States ---
    labels = ["p [m]", "f", "g", "h", "k", "L [rad]", "mass [kg]"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.flatten()
    for i in range(7):
        axes[i].plot(t_hr, x[i, :], "b-", linewidth=1.5)
        axes[i].set_xlabel("Time [hr]")
        axes[i].set_ylabel(labels[i])
        axes[i].set_title(labels[i])
        axes[i].grid(True)
    fig.suptitle("MEE State History — YAPSS LGL", fontsize=13)
    plt.tight_layout()

    # --- Controls ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 4))
    ctrl_labels = [r"$u_r$", r"$u_\theta$", r"$u_h$"]
    for i in range(3):
        axes2[i].plot(tc_hr, u[i, :], "r-", linewidth=1.2)
        axes2[i].set_xlabel("Time [hr]")
        axes2[i].set_ylabel(ctrl_labels[i])
        axes2[i].set_ylim([-1.1, 1.1])
        axes2[i].set_title(f"Control: {ctrl_labels[i]}")
        axes2[i].grid(True)
    fig2.suptitle("Control History — YAPSS LGL", fontsize=13)
    plt.tight_layout()

    # --- 3D trajectory ---
    def mee_to_eci(p_, f_, g_, h_, k_, L_):
        q  = 1 + f_*np.cos(L_) + g_*np.sin(L_)
        r  = p_ / q
        s2 = 1 + h_**2 + k_**2
        a2 = h_**2 - k_**2
        rX = (r/s2)*(np.cos(L_)+a2*np.cos(L_)+2*h_*k_*np.sin(L_))
        rY = (r/s2)*(np.sin(L_)-a2*np.sin(L_)+2*h_*k_*np.cos(L_))
        rZ = (2*r/s2)*(h_*np.sin(L_)-k_*np.cos(L_))
        return rX, rY, rZ

    rX, rY, rZ = mee_to_eci(x[0], x[1], x[2], x[3], x[4], x[5])

    fig3 = plt.figure(figsize=(8, 7))
    ax3  = fig3.add_subplot(111, projection="3d")
    ax3.plot(rX/1e6, rY/1e6, rZ/1e6, "r-", linewidth=1.5)
    ax3.scatter(*[v/1e6 for v in (rX[0], rY[0], rZ[0])],
                color="g", s=60, zorder=5, label="Initial")
    ax3.scatter(*[v/1e6 for v in (rX[-1], rY[-1], rZ[-1])],
                color="r", s=60, marker="s", zorder=5, label="Final")
    ax3.set_xlabel("x [Mm]"); ax3.set_ylabel("y [Mm]"); ax3.set_zlabel("z [Mm]")
    ax3.set_title("Optimal Low-Thrust Transfer (3D) — YAPSS")
    ax3.legend()

    # --- Hamiltonian (should be constant = 0 for free-time problem) ---
    fig4, ax4 = plt.subplots(figsize=(9, 3))
    ax4.plot(tc_hr, solution.phase[0].hamiltonian, "k-", linewidth=1.2)
    ax4.set_xlabel("Time [hr]")
    ax4.set_ylabel(r"Hamiltonian $\mathcal{H}$")
    ax4.set_title("Hamiltonian (should be constant ≈ 0 for free tf)")
    ax4.grid(True)
    plt.tight_layout()

    plt.show()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Solve and plot the low-thrust orbit transfer problem with YAPSS."""
    import time

    problem  = setup()
    t0       = time.perf_counter()
    solution = problem.solve()
    elapsed  = time.perf_counter() - t0

    print(f"\nTotal wall time: {elapsed:.2f} s")
    print(f"IPOPT status:   {solution.nlp_info.ipopt_status_message}")

    plot_solution(solution)


if __name__ == "__main__":
    main()