import numpy as np
from scipy.integrate import solve_ivp


def solve_u(psi_d, u0, ell, E, t_span, t_eval=None, sign_du0=1, **kwargs):
    """Solve (du/dt)^2 = (2E - ell^2/u^2) / (1 + psi_d(u))^2, u(0) = u0.

    Differentiating both sides gives the equivalent second-order ODE
        u'' = F'(u) / 2,  F(u) = (2E - ell^2/u^2) / (1 + psi_d(u))^2,
    which is integrated with scipy.

    Parameters
    ----------
    psi_d    : callable, derivative psi'(u) — any Python callable.
    u0       : float, initial value u(0).
    ell      : float, angular-momentum parameter.
    E        : float, energy parameter.
    t_span   : (t0, tf), integration interval.
    t_eval   : array-like or None, times at which to store the solution.
    sign_du0 : +1 or -1, sign of du/dt at t=0 (default +1).
    **kwargs : forwarded to scipy.integrate.solve_ivp (e.g. method, rtol, atol).

    Returns
    -------
    OdeResult from scipy.integrate.solve_ivp.
        sol.t      — time points
        sol.y[0]   — u(t)
        sol.y[1]   — u'(t)
        sol.sol(t) — dense interpolant (requires dense_output=True, set by default)
    """
    def F(u):
        return (2 * E - ell**2 / u**2) / (1 + psi_d(u))**2

    def dF(u):
        h = 1e-7 * abs(u) + 1e-12
        return (F(u + h) - F(u - h)) / (2 * h)

    def rhs(t, y):
        u, du = y
        return [du, dF(u) / 2]

    du0 = sign_du0 * np.sqrt(max(F(u0), 0.0))

    kwargs.setdefault("dense_output", True)
    return solve_ivp(rhs, t_span, [u0, du0], t_eval=t_eval, **kwargs)


def solve_v(sol_u, ell, v0, t_eval=None, **kwargs):
    """Solve dv/dt = ell / u(t)^2, v(0) = v0, given the solution u(t) from solve_u.

    Parameters
    ----------
    sol_u  : OdeResult returned by solve_u (must have dense_output=True).
    ell    : float, angular-momentum parameter.
    v0     : float, initial value v(t0).
    t_eval : array-like or None, times at which to store the solution.
    **kwargs : forwarded to scipy.integrate.solve_ivp.

    Returns
    -------
    OdeResult from scipy.integrate.solve_ivp.
        sol.t    — time points
        sol.y[0] — v(t)
    """
    t_span = (sol_u.t[0], sol_u.t[-1])

    def rhs(t, y):
        u = sol_u.sol(t)[0]
        return [ell / u**2]

    kwargs.setdefault("dense_output", True)
    return solve_ivp(rhs, t_span, [v0], t_eval=t_eval, **kwargs)
