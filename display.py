import numpy as np
import plotly.graph_objects as go

from solver import solve_u, solve_v


def display_surface(psi, a1, a2, b1, b2, num_u=60, num_v=60):
    """Display the image of [a1,a2]x[b1,b2] under (u,v) -> (u*cos(v), u*sin(v), psi(u)).

    Renders an interactive 3D plot rotatable with the mouse in a Jupyter notebook.
    psi can be any Python callable; it need not be numpy-vectorized.
    """
    _show([_surface_trace(psi, a1, a2, b1, b2, num_u, num_v)])


def _surface_trace(psi, a1, a2, b1, b2, num_u=60, num_v=60):
    u = np.linspace(a1, a2, num_u)
    v = np.linspace(b1, b2, num_v)
    U, V = np.meshgrid(u, v)
    return go.Surface(
        x=U * np.cos(V),
        y=U * np.sin(V),
        z=np.vectorize(psi)(U),
        colorscale="Viridis",
        showscale=False,
        opacity=0.7,
    )


def _curve_trace(sol_u, sol_v, psi, num_t=500):
    t0 = max(sol_u.t[0], sol_v.t[0])
    tf = min(sol_u.t[-1], sol_v.t[-1])
    t = np.linspace(t0, tf, num_t)
    u = sol_u.sol(t)[0]
    v = sol_v.sol(t)[0]
    return go.Scatter3d(
        x=u * np.cos(v),
        y=u * np.sin(v),
        z=np.vectorize(psi)(u),
        mode="lines",
        line=dict(width=4, color="red"),
    )


def _show(traces):
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.show()


def display_surface_and_curve(psi, psi_d, a1, a2, b1, b2,
                               ell, E, u0, v0, t_span,
                               num_u=60, num_v=60, num_t=200,
                               interactive=False,
                               ell_range=None, E_range=None,
                               u0_range=None, v0_range=None,
                               n_steps=10,
                               **kwargs):
    """Display the surface and the geodesic curve together in a single interactive plot.

    Internally calls solve_u and solve_v; extra keyword arguments are forwarded to them.

    Parameters
    ----------
    psi         : callable, ψ(u) — height function defining the surface.
    psi_d       : callable, ψ'(u) — derivative of ψ, required by solve_u.
    a1, a2      : float, interval for u used to draw the surface.
    b1, b2      : float, interval for v used to draw the surface.
    ell         : float, initial angular-momentum parameter.
    E           : float, initial energy parameter.
    u0          : float, initial value u(0).
    v0          : float, initial value v(0).
    t_span      : (t0, tf), integration interval.
    interactive : bool, if True add plotly sliders for ell, E, u0, v0.
    ell_range   : (min, max) for the ell slider; defaults to [ell/5, ell*5].
    E_range     : (min, max) for the E slider; defaults to [E/5, E*5].
    u0_range    : (min, max) for the u0 slider; defaults to [u0/5, u0*5].
    v0_range    : (min, max) for the v0 slider; defaults to [v0-pi, v0+pi].
    n_steps     : number of discrete steps per slider (default 10).
    """
    sol_u = solve_u(psi_d, u0, ell, E, t_span, **kwargs)
    sol_v = solve_v(sol_u, ell, v0, **kwargs)

    if not interactive:
        _show([
            _surface_trace(psi, a1, a2, b1, b2, num_u, num_v),
            _curve_trace(sol_u, sol_v, psi, num_t),
        ])
        return

    if ell_range is None:
        ell_range = (ell / 5, ell * 5)
    if E_range is None:
        E_range = (E / 5, E * 5)
    if u0_range is None:
        u0_range = (u0 / 5, u0 * 5)
    if v0_range is None:
        v0_range = (v0 - np.pi, v0 + np.pi)

    N = n_steps
    param_labels  = ["ℓ", "E", "u₀", "v₀"]
    param_grids   = [
        np.linspace(ell_range[0], ell_range[1], N),
        np.linspace(E_range[0],   E_range[1],   N),
        np.linspace(u0_range[0],  u0_range[1],  N),
        np.linspace(v0_range[0],  v0_range[1],  N),
    ]
    init_vals = [ell, E, u0, v0]
    active = [int(np.argmin(np.abs(g - v))) for g, v in zip(param_grids, init_vals)]

    def safe_curve(ell_v, E_v, u0_v, v0_v):
        try:
            s_u = solve_u(psi_d, u0_v, ell_v, E_v, t_span, **kwargs)
            s_v = solve_v(s_u, ell_v, v0_v, **kwargs)
            return _curve_trace(s_u, s_v, psi, num_t)
        except Exception:
            return go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                line=dict(width=4, color="red"), showlegend=False)

    # One group of N curves per parameter (others held at initial value).
    # Layout: [surface] [N ell-curves] [N E-curves] [N u0-curves] [N v0-curves]
    surface = _surface_trace(psi, a1, a2, b1, b2, num_u, num_v)
    groups = [
        [safe_curve(v, E,   u0,  v0)  for v in param_grids[0]],
        [safe_curve(ell, v, u0,  v0)  for v in param_grids[1]],
        [safe_curve(ell, E, v,   v0)  for v in param_grids[2]],
        [safe_curve(ell, E, u0,  v)   for v in param_grids[3]],
    ]

    all_traces = [surface]
    for g in groups:
        for tr in g:
            tr.visible = False
            all_traces.append(tr)

    # Initially show the curve for the active ell step.
    all_traces[1 + active[0]].visible = True

    def visibility(group_idx, step_idx):
        vis = [True]                        # surface always on
        for g in range(4):
            for s in range(N):
                vis.append(g == group_idx and s == step_idx)
        return vis

    sliders = []
    for g, (label, grid, act) in enumerate(zip(param_labels, param_grids, active)):
        steps = [
            {"method": "update",
             "label": f"{v:.3g}",
             "args": [{"visible": visibility(g, i)}]}
            for i, v in enumerate(grid)
        ]
        sliders.append({
            "active": act,
            "currentvalue": {"prefix": f"{label} = ", "xanchor": "center"},
            "pad": {"b": 10},
            "steps": steps,
        })

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=0, b=200),
        sliders=sliders,
    )
    fig.show()


def display_curve(sol_u, sol_v, psi, num_t=500):
    """Display the curve t -> (u*cos(v), u*sin(v), psi(u)) from solve_u/solve_v outputs.

    Uses the dense interpolants of sol_u and sol_v, evaluated on a common time grid
    spanning the intersection of their integration intervals.
    """
    _show([_curve_trace(sol_u, sol_v, psi, num_t)])
