import numpy as np

def R1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                      [0, c, s],
                      [0, -s, c]])

def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0],
                      [-s, c, 0],
                      [0, 0, 1]])

def HN_matrix(Omega, i, theta):
    return R3(theta) @ R1(i) @ R3(Omega)

def orbit_state(r, Omega, i, theta, thetadot):
    HN = HN_matrix(Omega, i, theta)
    NH = HN.T
    H_r    = np.array([r, 0.0, 0.0])
    H_rdot = np.array([0.0, r*thetadot, 0.0])
    return NH @ H_r, NH @ H_rdot

mu = 42828.3
R_mars = 3396.19

# ---- LMO ----
r_LMO = R_mars + 400.0
Omega_LMO = np.radians(20.0)
i_LMO = np.radians(30.0)
theta0_LMO = np.radians(60.0)
thetadot_LMO = np.sqrt(mu / r_LMO**3)

# ---- GMO ----
r_GMO = 20424.2
Omega_GMO = np.radians(0.0)
i_GMO = np.radians(0.0)
theta0_GMO = np.radians(250.0)
thetadot_GMO = np.sqrt(mu / r_GMO**3)


def RcN(t):
    theta_LMO_t = theta0_LMO + thetadot_LMO * t
    theta_GMO_t = theta0_GMO + thetadot_GMO * t

    N_r_LMO, N_rdot_LMO = orbit_state(r_LMO, Omega_LMO, i_LMO, theta_LMO_t, thetadot_LMO)
    N_r_GMO, N_rdot_GMO = orbit_state(r_GMO, Omega_GMO, i_GMO, theta_GMO_t, thetadot_GMO)

    delta_r = N_r_GMO - N_r_LMO
    r1 = -delta_r/np.linalg.norm(delta_r)
    r2 = np.cross(-r1, np.array([0,0,1]))/np.linalg.norm(np.cross(-r1, np.array([0,0,1])))
    r3 = np.cross(r1, r2)
    return np.array([r1, r2, r3])


def clean(matrix, tol=1e-10):
    m = matrix.copy()
    m[np.abs(m) < tol] = 0.0
    return m


def shadow_if_needed(sigma):
    """Enforce the short-rotation MRP set (|sigma| <= 1)."""
    if np.linalg.norm(sigma) > 1:
        sigma = -sigma / np.dot(sigma, sigma)
    return sigma


def dcm_mrp(R):
    """Robust DCM -> MRP conversion using Sheppard's method."""
    tr = R[0,0] + R[1,1] + R[2,2]

    b0_sq = (1 + tr) / 4
    b1_sq = (1 + 2*R[0,0] - tr) / 4
    b2_sq = (1 + 2*R[1,1] - tr) / 4
    b3_sq = (1 + 2*R[2,2] - tr) / 4

    sq = np.array([b0_sq, b1_sq, b2_sq, b3_sq])
    idx = np.argmax(sq)

    if idx == 0:
        b0 = np.sqrt(b0_sq)
        b1 = (R[1,2]-R[2,1])/(4*b0)
        b2 = (R[2,0]-R[0,2])/(4*b0)
        b3 = (R[0,1]-R[1,0])/(4*b0)
    elif idx == 1:
        b1 = np.sqrt(b1_sq)
        b0 = (R[1,2]-R[2,1])/(4*b1)
        b2 = (R[0,1]+R[1,0])/(4*b1)
        b3 = (R[2,0]+R[0,2])/(4*b1)
    elif idx == 2:
        b2 = np.sqrt(b2_sq)
        b0 = (R[2,0]-R[0,2])/(4*b2)
        b1 = (R[0,1]+R[1,0])/(4*b2)
        b3 = (R[1,2]+R[2,1])/(4*b2)
    else:
        b3 = np.sqrt(b3_sq)
        b0 = (R[0,1]-R[1,0])/(4*b3)
        b1 = (R[2,0]+R[0,2])/(4*b3)
        b2 = (R[1,2]+R[2,1])/(4*b3)

    if b0 < 0:
        b0, b1, b2, b3 = -b0, -b1, -b2, -b3

    sigma = np.array([b1, b2, b3]) / (1 + b0)
    return shadow_if_needed(sigma)


def mrp_subtract(sigma_BN, sigma_RN):
    """Returns sigma_B/R given sigma_B/N and sigma_R/N."""
    s1 = sigma_BN
    s2 = sigma_RN
    n1sq = np.dot(s1, s1)
    n2sq = np.dot(s2, s2)
    num = (1 - n2sq)*s1 - (1 - n1sq)*s2 + 2*np.cross(s1, s2)
    den = 1 + n1sq*n2sq + 2*np.dot(s1, s2)
    sigma_BR = num / den
    return shadow_if_needed(sigma_BR)


def skew(s):
    return np.array([[0, -s[2], s[1]],
                      [s[2], 0, -s[0]],
                      [-s[1], s[0], 0]])


def mrp_to_dcm(sigma):
    s_tilde = skew(sigma)
    s_sq = np.dot(sigma, sigma)
    BN = np.eye(3) + (8*(s_tilde @ s_tilde) - 4*(1-s_sq)*s_tilde) / (1+s_sq)**2
    return BN


def attitude_error(sigma_BN, omega_BN_B, RN, omega_RN_N):
    """Returns (sigma_B/R, omega_B/R)."""
    BN = mrp_to_dcm(sigma_BN)
    sigma_RN = dcm_mrp(RN)
    sigma_BR = mrp_subtract(sigma_BN, sigma_RN)
    BR = BN @ RN.T
    omega_BR_B = omega_BN_B - BN @ omega_RN_N
    return sigma_BR, omega_BR_B


def sigma_dot(sigma, omega):
    s_sq = np.linalg.norm(sigma)**2
    B_matrix = (1-s_sq)*np.eye(3) + 2*skew(sigma) + 2*np.outer(sigma,sigma)
    return 0.25 * (B_matrix @ omega)


I = np.diag([10.0, 5.0, 7.5])
I_inv = np.diag([1/10.0, 1/5.0, 1/7.5])

def omega_dot(omega, u):
    return I_inv@(-skew(omega)@I@omega+u)


def state_derivative(X, u):
    sigma = X[0:3]
    omega = X[3:6]
    return np.concatenate([sigma_dot(sigma, omega), omega_dot(omega, u)])


def rk4_step(X, u, dt):
    k1 = state_derivative(X,u)
    k2 = state_derivative(X+dt/2*k1,u)
    k3 = state_derivative(X+dt/2*k2,u)
    k4 = state_derivative(X+dt*k3,u)
    X_new = X + dt/6 * (k1+2*k2+2*k3+k4)
    X_new = np.concatenate([shadow_if_needed(X_new[0:3]), X_new[3:6]])
    return X_new


def propagate(X0, u_func, t0, tf, dt=1.0):
    n_steps = int((tf-t0)/dt)
    X = X0.copy()
    t = t0
    for _ in range(n_steps):
        u = u_func(t, X)
        X = rk4_step(X, u, dt)
        t += dt
    return X

def RnN(t):
    theta_LMO_t = theta0_LMO+thetadot_LMO*t
    HN_t = HN_matrix(Omega_LMO,i_LMO,theta_LMO_t)
    return RnH @ HN_t

def omega_RnN_N(t):
    HN_t = HN_matrix(Omega_LMO, i_LMO, theta0_LMO + thetadot_LMO*t)
    ih_N = HN_t[2]
    return thetadot_LMO * ih_N

def omega_RcN_N(t, dt=0.01):
    RcNdot_t = (RcN(t+dt) - RcN(t-dt)) / (2*dt)
    w_tilde = -RcN(t).T @ RcNdot_t
    return np.array([w_tilde[2,1], w_tilde[0,2], w_tilde[1,0]])

def u_nadirpointing(t, X):
    sigma_BN = X[0:3]
    omega_BN = X[3:6]
    RnN_t = RnN(t)
    omega_RnN_t = omega_RnN_N(t)
    sigma_BR, omega_BR = attitude_error(sigma_BN, omega_BN, RnN_t, omega_RnN_t)
    return -K*sigma_BR - P*omega_BR

def u_gmopointing(t,X):
    sigma_BN = X[0:3]
    omega_BN = X[3:6]
    RcN_t = RcN(t)
    omega_RcN_t = omega_RcN_N(t)
    sigma_BR, omega_BR = attitude_error(sigma_BN, omega_BN, RcN_t, omega_RcN_t)
    return -K*sigma_BR - P*omega_BR

def angle_between(v1,v2) :
    a = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return a

def pick_mode(t):
    theta_LMO_t = theta0_LMO + thetadot_LMO*t
    theta_GMO_t = theta0_GMO + thetadot_GMO*t
    N_r_LMO, _ = orbit_state(r_LMO, Omega_LMO, i_LMO, theta_LMO_t, thetadot_LMO)
    N_r_GMO, _ = orbit_state(r_GMO, Omega_GMO, i_GMO, theta_GMO_t, thetadot_GMO)

    if N_r_LMO[1] > 0:
        return "sun"
    elif np.degrees(angle_between(N_r_LMO, N_r_GMO)) < 35:
        return "comm"
    else:
        return "nadir"

def u_mission(t, X):
    mode = pick_mode(t)
    sigma_BN = X[0:3]
    omega_BN = X[3:6]

    if mode == "sun":
        sigma_BR, omega_BR = attitude_error(sigma_BN, omega_BN, RsN, omega_RsN_N0)
    elif mode == "nadir":
        sigma_BR, omega_BR = attitude_error(sigma_BN, omega_BN, RnN(t), omega_RnN_N(t))
    else:  # "comm"
        sigma_BR, omega_BR = attitude_error(sigma_BN, omega_BN, RcN(t), omega_RcN_N(t))

    return -K*sigma_BR - P*omega_BR
    
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=10, floatmode='fixed')

    # Task 6: attitude error validation at t0 (unchanged from before)
    sigma_BN_t0 = np.array([0.3, -0.4, 0.5])
    omega_BN_t0 = np.radians(np.array([1.00, 1.75, -2.20]))

    RsN = np.array([[-1,0,0],[0,0,1],[0,1,0]])
    omega_RsN_N0 = np.array([0.0, 0.0, 0.0])
    sigma_BRs, omega_BRs = attitude_error(sigma_BN_t0, omega_BN_t0, RsN, omega_RsN_N0)

    RnH = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    HN_0 = HN_matrix(Omega_LMO, i_LMO, theta0_LMO)
    RnN_0 = RnH @ HN_0
    ih_N_0 = HN_0[2,:]
    omega_RnN_N0 = thetadot_LMO * ih_N_0
    sigma_BRn, omega_BRn = attitude_error(sigma_BN_t0, omega_BN_t0, RnN_0, omega_RnN_N0)

    dt_fd = 0.01
    RcN_0 = RcN(0.0)
    RcNdot_0 = (RcN(dt_fd) - RcN(-dt_fd)) / (2*dt_fd)
    w_tilde_N = -RcN_0.T @ RcNdot_0
    omega_RcN_N0 = np.array([w_tilde_N[2,1], w_tilde_N[0,2], w_tilde_N[1,0]])
    sigma_BRc, omega_BRc = attitude_error(sigma_BN_t0, omega_BN_t0, RcN_0, omega_RcN_N0)

    sigma0 = np.array([0.3, -0.4, 0.5])
    omega0 = np.radians(np.array([1.00, 1.75, -2.20]))
    X0 = np.concatenate([sigma0, omega0])

    # ============================================================
    # TASK 8: Sun-pointing PD control
    # ============================================================
    P, K = 0.166666666666667, 0.005555555555556

    # ============================================================
    # FIX #2: this is the actual controller -- it computes the
    # REAL tracking error sigma_B/Rs, omega_B/Rs via
    # attitude_error(), NOT "mrp_to_dcm(sigma) @ sigma" (which
    # isn't a meaningful quantity in this problem).
    # ============================================================
    def u_sunpointing(t, X):
        sigma_BN = X[0:3]
        omega_BN = X[3:6]
        sigma_BR, omega_BR = attitude_error(sigma_BN, omega_BN, RsN, omega_RsN_N0)
        return -K*sigma_BR - P*omega_BR

    print("=== Task 8: Sun-pointing closed-loop control ===")
    for t_final in [15, 100, 200, 400]:
        X_final = propagate(X0, u_sunpointing, 0.0, t_final, dt=1.0)
        sigma_BN_final = X_final[0:3]
        omega_BN_final = X_final[3:6]
        sigma_BR_final, _ = attitude_error(sigma_BN_final, omega_BN_final, RsN, omega_RsN_N0)
        print(f"t={t_final:4d}s   sigma_B/N = {sigma_BN_final}   "
              f"| sigma_B/R (error) = {sigma_BR_final}")

    print("=== Task 9: Nadir-pointing closed-loop control ===")
    for t_final in [15, 100, 200, 400]:
        X_final = propagate(X0, u_nadirpointing, 0.0, t_final, dt=1.0)
        sigma_BN_final = X_final[0:3]
        omega_BN_final = X_final[3:6]
        sigma_BR_final, _ = attitude_error(sigma_BN_final, omega_BN_final, RnN(t_final), omega_RnN_N(t_final))
        print(f"t={t_final:4d}s   sigma_B/N = {sigma_BN_final}   "
            f"| sigma_B/R (error) = {sigma_BR_final}")

    print("=== Task 9: gmo-pointing closed-loop control ===")
    for t_final in [15, 100, 200, 400]:
        X_final = propagate(X0, u_gmopointing, 0.0, t_final, dt=1.0)
        sigma_BN_final = X_final[0:3]
        omega_BN_final = X_final[3:6]
        sigma_BR_final, _ = attitude_error(sigma_BN_final, omega_BN_final, RcN(t_final), omega_RcN_N(t_final))
        print(f"t={t_final:4d}s   sigma_B/N = {sigma_BN_final}   "
            f"| sigma_B/R (error) = {sigma_BR_final}")

    sigma0 = np.array([0.3, -0.4, 0.5])
    omega0 = np.radians(np.array([1.00, 1.75, -2.20]))
    X0 = np.concatenate([sigma0, omega0])

    for t_final in [300, 2100, 3400, 4400, 5600]:
        X_final = propagate(X0, u_mission, 0.0, t_final, dt=0.01)  # dt=0.01 given our Task 9 convergence lesson
        print(f"t={t_final}s  mode={pick_mode(t_final)}  sigma_B/N = {X_final[0:3]}")