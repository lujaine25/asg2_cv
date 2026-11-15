"""Part 2: fitting transforms (Q2.1) and RANSAC (Q2.2).

You fill in the blocks marked TODO. Everything else already runs.
Check your work with:  python check.py 2.1   and   python check.py 2.2
"""
import numpy as np, cv2
from utils import apply
from features import match


# ===================================================================== Q2.1 fitting
def fit_similarity(p, q):
    """Least-squares similarity (rotation, uniform scale, translation) taking p to q:
    x' = a x - b y + tx,  y' = b x + a y + ty.

    Args:
        p: (N, 2) points, N >= 2.
        q: (N, 2) matching points.

    Returns:
        (3, 3) transform [[a, -b, tx], [b, a, ty], [0, 0, 1]].
    """
    n = len(p)
    A, rhs = np.zeros((2 * n, 4)), np.zeros(2 * n)       # unknowns (a, b, tx, ty)
    # ---------------------------------------------------------------- TODO 2.1a ----
    # A similarity is rotation + scale + shift. Its rotation part is [[a, -b], [b, a]]
    # (a = s*cos(theta), b = s*sin(theta)), which gives:
    #     x' = a*x - b*y + tx
    #     y' = b*x + a*y + ty
    # Unknowns are (a, b, tx, ty), so each equation becomes one row of A.
    x, y = p[:, 0], p[:, 1]
    A[0::2] = np.column_stack([x, -y, np.ones(n), np.zeros(n)])   # x' equation
    A[1::2] = np.column_stack([y,  x, np.zeros(n), np.ones(n)])   # y' equation
    # Right-hand side is the target point...lstsq finds the best fit when matches are noisy.
    rhs[0::2], rhs[1::2] = q[:, 0], q[:, 1]
    # ---------------------------------------------------------------- end TODO -----
    a, b, tx, ty = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])


def fit_homography(p, q, normalize=True):
    """Homography taking p to q by the direct linear transform (DLT), solved with the SVD.

    Args:
        p: (N, 2) points, N >= 4.
        q: (N, 2) matching points.
        normalize: apply Hartley normalization to both point sets first.

    Returns:
        (3, 3) homography with H[2, 2] = 1.
    """
    Tp, Tq = np.eye(3), np.eye(3)
    if normalize:
        # ------------------------------------------------------------ TODO 2.1b ----
        # Center each point set at (0, 0) and scale it to an average distance of sqrt(2), so the DLT is well conditioned.
        def hartley(pts):
            c = pts.mean(axis=0)                          # centre of the points
            d = np.linalg.norm(pts - c, axis=1).mean()    # average distance from the centre
            s = np.sqrt(2) / d                            # scale that makes it sqrt(2)
            return np.array([[s, 0, -s * c[0]],           # subtract centre, then scale
                             [0, s, -s * c[1]],
                             [0, 0, 1.0]])
        Tp, Tq = hartley(p), hartley(q)                   # each set has its own centre and spread
        # ------------------------------------------------------------ end TODO -----
    pn, qn = apply(Tp, p), apply(Tq, q)
    # ---------------------------------------------------------------- TODO 2.1c ----
    # Each match gives two linear equations in the 9 entries of H; stack them all into A @ h = 0.
    m = len(pn)
    x, y = pn[:, 0], pn[:, 1]
    u, v = qn[:, 0], qn[:, 1]
    z, o = np.zeros(m), np.ones(m)                        # columns of 0s and 1s
    A = np.zeros((2 * m, 9))
    A[0::2] = np.column_stack([-x, -y, -o,  z,  z,  z, u * x, u * y, u])   # u equation
    A[1::2] = np.column_stack([ z,  z,  z, -x, -y, -o, v * x, v * y, v])   # v equation
    # Noise means A @ h is never exactly 0, so pick the unit h that makes |A h| smallest: the last row of Vt.
    Hn = np.linalg.svd(A)[2][-1].reshape(3, 3)
    # ---------------------------------------------------------------- end TODO -----
    H = np.linalg.inv(Tq) @ Hn @ Tp                       # undo the normalization
    return H / H[2, 2]


# ===================================================================== Q2.2 RANSAC
def plausible(T, homography):
    """Whether a camera flying over the ground could produce this transform.

    Args:
        T: (3, 3) transform.
        homography: True to also check the perspective terms.

    Returns:
        False if T is not finite, if its scale (sqrt|det| of the top-left 2 x 2 block) is outside 0.6 to 1.6,
        or, for a homography, if a perspective term |T[2, 0]| or |T[2, 1]| is above 0.002. True otherwise.
    """
    if not np.all(np.isfinite(T)) or abs(T[2, 2]) < 1e-12:
        return False
    T = T / T[2, 2]
    s = np.sqrt(abs(np.linalg.det(T[:2, :2])))
    if not 0.6 < s < 1.6:
        return False
    return not homography or np.abs(T[2, :2]).max() <= 2e-3


def ransac(p, q, fit, sample, thresh=3.0, p_success=0.99, max_iter=5000, seed=0):
    """RANSAC for any model.

    Args:
        p: (N, 2) points.
        q: (N, 2) matching points.
        fit: fit_similarity or fit_homography.
        sample: points per sample, 2 for a similarity and 4 for a homography.
        thresh: inlier threshold in pixels.
        p_success: probability of drawing at least one all-inlier sample.
        max_iter: upper bound on the iterations.
        seed: seed of the random generator.

    Returns:
        T: (3, 3) transform refitted on all its inliers, or None if no plausible fit was found.
        inliers: (N,) boolean mask, or None.
        iterations: how many iterations ran.
    """
    rng = np.random.default_rng(seed)
    n = len(p)
    if n < sample + 2:
        return None, None, 0
    best_inliers, needed, it = None, max_iter, 0
    while it < min(needed, max_iter):
        it += 1
        idx = rng.choice(n, sample, replace=False)
        try:
            T = fit(p[idx], q[idx])
        except np.linalg.LinAlgError:
            continue
        if not plausible(T, homography=(sample == 4)):
            continue
        # ------------------------------------------------------------ TODO 2.2 -----
        # Matches that T maps to within thresh pixels of their partner are inliers.
        proj = apply(T, p)
        inl = np.linalg.norm(proj - q, axis=1) < thresh
        k = int(inl.sum())                                # how many matches agree with T

        # Keep T if it has more inliers than the best so far.
        if k >= sample and (best_inliers is None or k > best_inliers.sum()):
            best_inliers = inl
            w = k / n                                     # inlier ratio
            ws = w ** sample                              # chance a random sample is all inliers
            if ws >= 1.0:
                needed = it                               # everything is an inlier, stop
            else:
                # Number of samples needed to get at least one all-inlier sample with probability p_success
                needed = int(np.ceil(np.log(1 - p_success) / np.log(1 - ws)))
        # ------------------------------------------------------------ end TODO -----
    if best_inliers is None:
        return None, None, it
    T = fit(p[best_inliers], q[best_inliers])                   # refit on all the inliers
    if not plausible(T, homography=(sample == 4)):
        return None, None, it
    return T / T[2, 2], best_inliers, it


def estimate(F1, F2, model='similarity', thresh=3.0):
    """Transform between two frames from their features: match(), then ransac().

    Args:
        F1: (keypoints, descriptors) of the first frame.
        F2: (keypoints, descriptors) of the second frame.
        model: 'similarity' or 'homography'.
        thresh: RANSAC inlier threshold in pixels.

    Returns:
        T: (3, 3) transform taking pixels of the first frame to pixels of the second, or None.
        n_inliers: number of inliers, 0 if T is None.
        inlier_points: (p, q), the inlier matches as two (n_inliers, 2) arrays, or None.
    """
    (k1, d1), (k2, d2) = F1, F2
    mm = match(d1, d2)
    if len(mm) < 8:
        return None, 0, None
    p, q = k1[mm[:, 0]], k2[mm[:, 1]]
    fit, s = (fit_similarity, 2) if model == 'similarity' else (fit_homography, 4)
    T, inl, _ = ransac(p, q, fit, s, thresh)
    if T is None:
        return None, 0, None
    return T, int(inl.sum()), (p[inl], q[inl])
