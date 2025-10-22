import numpy as np
from scipy.optimize import root_scalar


# This function propagates two body problem
def twobody_propagator(oe0: np.ndarray,
                       dt: float,
                       mu: float) -> np.ndarray:
    # Retrieve required orbital elements
    a = oe0[0]
    M0 = oe0[5]

    # Compute mean motion
    n = np.sqrt(mu/a**3)

    # Move forward
    M = M0 + n*dt

    # Remove elapsed orbits
    M = M % (2*np.pi)

    # Store new mean anomaly
    oe = np.zeros(6)
    oe[0:5] = oe0[0:5]
    oe[5] = M

    return oe


# This function transform Euler angles
# to direction cosine matrix
def eul2dcm(eul, sequence='313'):
    if sequence == '313':
        dcm1 = dcmz(eul[0])
        dcm2 = dcmx(eul[1])
        dcm3 = dcmz(eul[2])

    # Multiply in order
    dcm = dcm3 @ dcm2 @ dcm1

    return dcm


# This creates dcm for fundamental rotation in x
def dcmx(angle):
    # Cosine and sine
    c = np.cos(angle)
    s = np.sin(angle)

    # Compute dcm
    dcmx = np.array([[1, 0, 0],
                     [0, c, s],
                     [0, -s, c]])

    return dcmx


# This creates dcm for fundamental rotation in y
def dcmy(angle):
    # Cosine and sine
    c = np.cos(angle)
    s = np.sin(angle)

    # Compute dcm
    dcmy = np.array([[c, 0, -s],
                     [0, 1, 0],
                     [s, 0, c]])

    return dcmy


# This creates dcm for fundamental rotation in z
def dcmz(angle):
    # Cosine and sine
    c = np.cos(angle)
    s = np.sin(angle)

    # Compute dcm
    dcmz = np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])

    return dcmz

# This function transforms time
# from periapsis into mean anomaly
def t2M(t: float,
        n: float) -> float:
    # Substract the number of orbits
    T = 2*np.pi / n
    t = t % T

    # Compute mean anomaly
    M = n * t

    return M


# This function transforms mean anomaly
# into eccentric anomaly
def M2E(M: float,
        e: float = 0) -> float:
    # Kepler equation
    def kep_eq(E: float) -> float:
        f = M - (E - e*np.sin(E))

        return f

    # Call root solver
    sol = root_scalar(kep_eq,
                      x0=M,
                      xtol=1e-6,
                      rtol=1e-6,
                      method='newton')

    # Retrieve only the root
    E = sol.root

    return E


# This function transforms eccentric
# anomaly into true anomaly
def E2nu(E: float,
         e: float = 0) -> float:
    # Compute true anomaly
    nu = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2),
                        np.sqrt(1-e) * np.cos(E/2))

    return nu


# This function transforms time from periapsis
# into true anomaly
def t2nu(t: float,
         n: float,
         e: float = 0) -> float:
    # Compute mean anomaly
    M = t2M(t, n)

    # Compute eccentric anomaly
    E = M2E(M, e=e)

    # Compute true anomaly
    nu = E2nu(E, e=e)

    return nu


# This function transforms mean anomaly
# to time from periapsis
def M2t(M: float,
        n: float) -> float:
    # Substract the number of orbits
    M = M % (2*np.pi)

    # Compute time from periapsis
    t = M / n

    return t


# This function transforms mean anomaly
# to time from periapsis
def E2M(E: float,
        e: float) -> float:
    # Substract the number of orbits
    E = E % (2*np.pi)

    # Compute mean anomaly
    M = E - e*np.sin(E)

    return M


# This function transforms true anomaly
# to eccentric anomaly
def nu2E(nu: float,
         e: float = 0) -> float:
    # Compute eccentric anomaly
    E = 2 * np.arctan2(np.sqrt(1-e) * np.sin(nu/2),
                       np.sqrt(1+e) * np.cos(nu/2))

    return E


# This function transforms true anomaly
# to time from periapsis
def nu2t(nu: float,
         n:  float,
         e:  float = 0) -> float:
    # Compute eccentric anomaly
    E = M2E(nu, e=e)

    # Compute mean anomaly
    M = E2M(E, e=e)

    # Compute time from periapsis
    t = (M, n)

    return t


# This function transforms orbital elements
# to cartesian position and velocity
def oe2cartesian(oe: np.ndarray,
                 mu: float) -> np.ndarray:
    # Extract orbital elements
    a = oe[0]
    e = oe[1]
    inc = oe[2]
    RAAN = oe[3]
    omega = oe[4]
    M = oe[5]

    # Transform mean anomaly to true anomaly
    E = M2E(M, e=e)
    nu = E2nu(E, e=e)

    # Compute the conic parameter and r
    p = a * (1-e**2)
    r = p / (1+e*np.cos(nu))

    # Compute radial auxiliary vectors
    # for perifocal frame
    er = np.array([np.cos(nu),
                   np.sin(nu),
                   0])
    ev = np.array([-np.sin(nu),
                   e+np.cos(nu),
                   0])

    # Compute position and velocity in perifocal frame
    pos_P = r * er
    vel_P = np.sqrt(mu/p) * ev

    # Compute euler angles in 313 sequence
    eul313 = np.array([RAAN,
                       inc,
                       omega])
    dcm_PN = eul2dcm(eul313, sequence='313')
    dcm_NP = dcm_PN.T

    # Convert position and velocity to cartesian
    pos = dcm_NP @ pos_P
    vel = dcm_NP @ vel_P

    return pos, vel


# This function transforms cartesian position
# and velocity to orbital elements
def cartesian2oe(pos: np.ndarray,
                 vel: np.ndarray,
                 mu:  float) -> np.ndarray:
    # Orbital radius and velocity
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    # Specific angular momentum
    h = np.cross(pos, vel)
    h_norm = np.linalg.norm(h)

    # Eccentricity
    e_vec = (np.cross(vel,h) - (mu/r)*pos) / mu
    e = np.linalg.norm(e_vec)

    # Specific energy
    eps = v**2/2 - mu/r

    # Semi-major axis
    a = -mu / (2*eps)

    # True anomaly
    p = h_norm**2 / mu
    cos_nu = np.dot(e_vec, pos) / (e*r)
    #cos_nu = (p/r - 1) / e
    nu = np.arccos(np.clip(cos_nu, -1, 1))

    # Correct if r·v < 0
    if np.dot(pos, vel) < 0:
        nu = 2*np.pi - nu

    # Convert true to mean anomaly
    E = nu2E(nu, e)
    M = E2M(E, e)

    # Orbital inclination
    k_R = np.array([0, 0, 1])
    cos_inc = np.dot(h, k_R) / h_norm
    inc = np.arccos(np.clip(cos_inc, -1, 1))

    # Normal vector to reference plane
    n = np.cross(k_R, h)
    n_norm = np.linalg.norm(n)

    # Check special cases
    if n_norm != 0:
        n /= n_norm
    else:
        n = np.array([1, 0, 0])  # Equatorial orbit

    # RAAN
    cos_RAAN = n[0]
    RAAN = np.arccos(np.clip(cos_RAAN, -1, 1))
    if n[1] < 0:
        RAAN = 2*np.pi - RAAN

    # Periapsis argument
    cos_omega = np.dot(e_vec, n) / e
    omega = np.arccos(np.clip(cos_omega, -1, 1))
    if e_vec[2] < 0:
        omega = 2*np.pi - omega

    # Place all orbital elements in vector
    oe = np.array([a,
                   e,
                   inc,
                   RAAN,
                   omega,
                   M])

    return oe


# This function transform inertial coordinates
# to geographical ones
def inertial2geographical(pos_N:  np.ndarray,
                          dcm_PN: np.ndarray) -> np.ndarray:
    # Compute geographical coordinates
    pos_P = dcm_PN @ pos_N

    return pos_P
