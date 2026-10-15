"""
Homework 1: chain of N masses falling under gravity with springs and viscous drag (1D).

Same structure as in_class_10-7.py: implicit Euler time stepping with Newton's method.
Differences from the 2D falling-beam code:
  - 1D: one DOF per node, q = [x1, x2, ..., xN]   (positive x = downward)
  - springs are linear with stiffness k[i]; spring i joins node i-1 and node i,
    and node 0 is a fixed wall (so k[0] connects the wall to mass 1)
  - b[i] is the viscous drag coefficient of node i against the fluid (Fv = -b u)
  - no bending, so gradEb / hessEb are not needed, and the 1D spring energy is
    simple enough that we define its own gradEs / hessEs below
  - gravity W = m g (no buoyancy: the homework gives no fluid density)

Run:  python3 homework1_simulations.py
"""

import numpy as np

g = 9.8


# ---------------------------------------------------------------------------
# Spring energy: E = 0.5 * k * (x_{i} - x_{i-1})^2   (rest length 0)
# ---------------------------------------------------------------------------

def gradEs(xkm1, xk, k):
    """Gradient of E_s with respect to (x_{k-1}, x_k)."""
    d = xk - xkm1
    return np.array([-k * d, k * d])


def hessEs(xkm1, xk, k):
    """Hessian of E_s with respect to (x_{k-1}, x_k)."""
    return np.array([[k, -k],
                     [-k, k]])


# ---------------------------------------------------------------------------
# Elastic force: assemble over all springs
# ---------------------------------------------------------------------------

def getFs(q, kvec):
    """Spring force Fs = -dE_s/dq and its Jacobian Js = -d2E_s/dq2."""
    ndof = q.size
    Fs = np.zeros(ndof)
    Js = np.zeros((ndof, ndof))

    for i in range(ndof):  # spring i joins node i-1 and node i
        if i == 0:
            # wall (x = 0) to node 0: only node 0 has a DOF
            ind = np.array([0])
            xkm1, xk = 0.0, q[0]
            Fs[0] -= gradEs(xkm1, xk, kvec[0])[1]
            Js[0, 0] -= hessEs(xkm1, xk, kvec[0])[1, 1]
        else:
            ind = np.array([i - 1, i])
            Fs[ind] -= gradEs(q[i - 1], q[i], kvec[i])
            Js[np.ix_(ind, ind)] -= hessEs(q[i - 1], q[i], kvec[i])

    return Fs, Js


# ---------------------------------------------------------------------------
# One implicit Euler step (Newton's method)
# ---------------------------------------------------------------------------

def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat, kvec, W, C):
    """Solve for q_new. Returns (q_new, flag); flag = -1 if Newton did not converge."""
    q_new = q_guess.copy()
    iter_count = 0
    error = tol * 10

    while error > tol:
        Fs, Js = getFs(q_new, kvec)

        # Viscous force
        Fv = -C @ (q_new - q_old) / dt
        Jv = -C / dt

        # Equations of motion and Jacobian
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fs + W + Fv)
        J = mMat / dt**2 - (Js + Jv)

        q_new = q_new - np.linalg.solve(J, f)
        error = np.linalg.norm(f)

        iter_count += 1
        if iter_count > maximum_iter:
            return q_new, -1

    return q_new, 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(m, kvec, bvec, dt, totalTime, scheme='implicit'):
    """
    Simulate N masses (1D). Returns t, q_hist, u_hist (histories include t = 0).
    scheme = 'implicit' (backward Euler + Newton) or 'explicit' (forward Euler).
    """
    m = np.asarray(m, float)
    kvec = np.asarray(kvec, float)
    bvec = np.asarray(bvec, float)
    ndof = m.size

    mMat = np.diag(m)
    W = m * g                      # gravity, positive = downward
    C = np.diag(bvec)              # viscous drag

    maximum_iter = 100
    tol = 1e-6   # residual is in newtons; roundoff in m*q/dt^2 is ~1e-10 at dt = 1e-3

    # Initial conditions: at rest at x = 0
    q0 = np.zeros(ndof)
    u = np.zeros(ndof)

    Nsteps = round(totalTime / dt)
    t = np.arange(Nsteps + 1) * dt
    q_hist = np.zeros((Nsteps + 1, ndof))
    u_hist = np.zeros((Nsteps + 1, ndof))

    for timeStep in range(1, Nsteps + 1):
        if scheme == 'implicit':
            q, flag = objfun(q0, q0, u, dt, tol, maximum_iter,
                             m, mMat, kvec, W, C)
            if flag < 0:
                raise RuntimeError(f'Newton did not converge at t = {t[timeStep]:.4f} s')
            u = (q - q0) / dt
        else:  # explicit (forward) Euler: uses OLD q0 and u
            Fs, _ = getFs(q0, kvec)
            a = (Fs + W - C @ u) / m
            q = q0 + dt * u
            u = u + dt * a

        q0 = q
        q_hist[timeStep] = q
        u_hist[timeStep] = u

    return t, q_hist, u_hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Part A: free fall (k = 0, b = 0)
    t, q, u = simulate([1], [0], [0], 1e-3, 1)
    print(f'Q1:  x1(1 s)           = {q[-1, 0]:.4f} m   (exact 0.5 g t^2 = 4.9)')

    t, q, u = simulate([1, 2, 3], [0, 0, 0], [0, 0, 0], 1e-3, 2)
    print(f'Q2:  x3(2 s)           = {q[-1, 2]:.4f} m   (exact 19.6, independent of mass)')

    # Part B: terminal velocity (k = 0, b > 0), u = m g / b
    t, q, u = simulate([1], [0], [2], 1e-3, 20)
    print(f'Q3:  u1(20 s)          = {u[-1, 0]:.4f} m/s (m g / b = 4.9)')

    t, q, u = simulate([1, 1], [0, 0], [2, 2], 1e-3, 20)
    print(f'Q4:  u2(20 s)          = {u[-1, 1]:.4f} m/s (m g / b = 4.9)')

    # Part C: static equilibrium
    t, q, u = simulate([1], [10], [1], 1e-3, 50)
    print(f'Q5:  x1(50 s)          = {q[-1, 0]:.4f} m   (m g / k = 0.98)')

    t, q, u = simulate([1]*5, [10]*5, [5]*5, 1e-2, 100)
    print(f'Q6:  x5(100 s)         = {q[-1, 4]:.4f} m   (sum of stretches = 14.7)')

    t, q, u = simulate([1, 2, 3], [10, 20, 30], [5, 10, 15], 1e-2, 100)
    print(f'Q7:  x3(100 s)         = {q[-1, 2]:.4f} m   (5.88 + 2.45 + 0.98 = 9.31)')

    # Part D: artificial damping in implicit Euler (m = k = 1, b = 0)
    t, q, u = simulate([1], [1], [0], 0.1, 100)
    print(f'Q8:  x1(100 s), dt=0.1 = {q[-1, 0]:.4f} m   (exact would be {9.8*(1-np.cos(100)):.4f})')

    for dt in (0.1, 0.01):
        t, q, u = simulate([1], [1], [0], dt, 100)
        w = q[t >= 90, 0]
        print(f'Q9:  dt = {dt:<5}: oscillation amplitude near t=100 = {(w.max() - w.min())/2:.4f} m')

    # Part E: explicit Euler (m = k = 1, b = 0, dt = 0.1)
    t, q, u = simulate([1], [1], [0], 0.1, 100, scheme='explicit')
    for lo, hi in [(0, 10), (40, 50), (90, 100)]:
        w = q[(t >= lo) & (t <= hi), 0]
        print(f'Q10: amplitude over [{lo},{hi}] s = {(w.max() - w.min())/2:.2f} m (grows)')

    mask = t <= 10
    i = q[mask, 0].argmax()
    print(f'Q11: max x1 on [0, 10 s] = {q[mask, 0][i]:.4f} m at t = {t[mask][i]:.1f} s')