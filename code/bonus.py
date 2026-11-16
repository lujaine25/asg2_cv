"""Bonus: fix the camera. Run it from code/ once Parts 1 to 3 work:

    python bonus.py

The camera of loop and lost_at_sea has radial lens distortion. We model it as
    p_d = c + (p - c) * (1 + kappa * |p - c|^2 / 600^2)
(p: where an ideal camera puts a point, p_d: where it appears in the frame, c: the image center),
estimate kappa from our own matches only (no ground truth), undistort the frames, and repeat Q3.2.
The ground truth is used only to score the results at the end.

Writes to results/bonus/:
    kappa.md                     the kappa found for every flight
    kappa_curves.png             how well a homography fits the matches, against kappa
    models.md                    Q3.2 again: similarity vs homography, before and after undistorting
    paths_loop.png               both models on loop after undistorting, against the truth
    error_vs_radius.png          the Q3.2 match-error plot, after undistorting
    undistort_loop.jpg           a loop frame before and after undistorting
    sensitivity_loop.md          loop chained with kappas around ours: how much the answer depends on kappa
"""
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils, gt, viz
from features import features, match
from fitting import estimate
from odometry import chain

S = 600.0                                   # the 600 of the distortion model (a focal length, in pixels)


# ===================================================================== the distortion model
def distort(p, kappa, c):
    """Ideal points -> where the real camera puts them: p_d = c + (p - c)(1 + kappa r^2 / 600^2)."""
    d = p - c
    r2 = (d ** 2).sum(1, keepdims=True)
    return c + d * (1 + kappa * r2 / S ** 2)


def undistort_points(pd, kappa, c, iters=20):
    """Distorted points -> ideal points: invert distort() by fixed-point iteration.

    p_d - c = (p - c) * f(p), so p = c + (p_d - c) / f(p). Start at p = p_d and repeat; f is close to 1,
    so this converges in a few steps.
    """
    p = pd.copy()
    for _ in range(iters):
        r2 = ((p - c) ** 2).sum(1, keepdims=True)
        p = c + (pd - c) / (1 + kappa * r2 / S ** 2)
    return p


def undistort_image(img, kappa):
    """Resample a frame as an ideal camera would have seen it.

    For every pixel p of the output (ideal) image, distort() says where that point is in the real frame,
    and cv2.remap() reads it there. Also returns the mask of output pixels that came from inside the frame.
    """
    h, w = img.shape[:2]
    c = np.array([w / 2, h / 2])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pd = distort(np.c_[xx.ravel(), yy.ravel()], kappa, c)
    mx = pd[:, 0].reshape(h, w).astype(np.float32)
    my = pd[:, 1].reshape(h, w).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return out, valid


def features_undistorted(img, kappa):
    """Our Part 1 features on the undistorted frame, without corners whose patch reaches outside the frame."""
    und, valid = undistort_image(img, kappa)
    kps, desc = features(und)
    if valid.all():
        return kps, desc
    ok = cv2.erode(valid.astype(np.uint8), np.ones((61, 61), np.uint8))   # keep 30 px away from invalid pixels
    keep = ok[kps[:, 1].astype(int), kps[:, 0].astype(int)] > 0
    return kps[keep], desc[keep]


# ===================================================================== estimating kappa from our matches
def collect_matches(F, gaps=(2, 4, 6, 8), thresh=6.0):
    """Inlier matches between frames k and k + gap, found with our features and RANSAC (homography).

    Gaps of several frames, not 1: the more the ground moves between two frames, the more differently the
    distortion bends the same point in each, so the easier it is to measure. Frames 8 apart still overlap
    by more than half. The threshold is looser than 3 px because, before undistorting, the distortion itself
    pushes correct matches near the border a few pixels off.
    """
    pairs = []
    with np.errstate(all='ignore'):                   # degenerate RANSAC samples are skipped anyway
        for g in gaps:
            for k in range(0, len(F) - g):
                H, n, pts = estimate(F[k], F[k + g], 'homography', thresh)
                if H is not None and n >= 30:
                    pairs.append(pts)
    return pairs


def fit_homography_fast(p, q):
    """The same normalized DLT as fit_homography() in fitting.py, only faster: the unit h that makes |Ah|
    smallest is the eigenvector of the 9 x 9 matrix A^T A with the smallest eigenvalue, the same vector as
    the last row of Vt in the SVD of A. This avoids the large U matrix of a full SVD, which matters here
    because we fit tens of thousands of homographies."""
    def hartley(x):
        c = x.mean(0)
        s = np.sqrt(2) / np.linalg.norm(x - c, axis=1).mean()
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    Tp, Tq = hartley(p), hartley(q)
    pn, qn = utils.apply(Tp, p), utils.apply(Tq, q)
    x, y, u, v = pn[:, 0], pn[:, 1], qn[:, 0], qn[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.r_[np.c_[-x, -y, -o, z, z, z, u * x, u * y, u],
              np.c_[z, z, z, -x, -y, -o, v * x, v * y, v]]
    h = np.linalg.eigh(A.T @ A)[1][:, 0]             # eigenvalues come sorted, smallest first
    H = np.linalg.inv(Tq) @ h.reshape(3, 3) @ Tp
    return H / H[2, 2]


def fit_cost(pairs, kappa, c):
    """How badly a homography fits the matches once they are undistorted with this kappa.

    For each pair of frames: undistort both point sets, fit a homography by least squares, and take the mean
    distance from H(p) to q. The cost is the average over all pairs. The ground is nearly flat, so the frames
    of an ideal camera are related exactly by a homography: the right kappa gives the lowest cost.
    We use the mean, not the median, because the distortion is strongest at the border, and the median
    would mostly listen to the points near the center, where there is little distortion to see.
    """
    costs = []
    for p, q in pairs:
        pu, qu = undistort_points(p, kappa, c), undistort_points(q, kappa, c)
        H = fit_homography_fast(pu, qu)
        costs.append(np.mean(np.linalg.norm(utils.apply(H, pu) - qu, axis=1)))
    return float(np.mean(costs))


def estimate_kappa(F, frame_size):
    """Find kappa by a grid search: a coarse grid, then a finer one around its best value.

    Returns:
        kappa, and (grid, cost on the grid, cost at kappa = 0, cost at kappa) for the plot and the table.
    """
    c = np.array(frame_size) / 2.0
    pairs = collect_matches(F)
    grid = np.linspace(-0.4, 0.4, 33)                 # steps of 0.025
    coarse = [fit_cost(pairs, k, c) for k in grid]
    k0 = grid[int(np.argmin(coarse))]
    fine = np.linspace(k0 - 0.025, k0 + 0.025, 11)    # steps of 0.005
    cost = [fit_cost(pairs, k, c) for k in fine]
    kappa = float(fine[int(np.argmin(cost))])
    cost0 = fit_cost(pairs, 0.0, c)
    # Only believe the distortion if it explains the matches noticeably better than an ideal camera. If the
    # best kappa improves the fit by less than 2 %, the cost curve is flat and its minimum is just noise.
    if (cost0 - min(cost)) / cost0 < 0.02:
        kappa = 0.0
    return kappa, (grid, coarse, cost0, min(cost))


# ===================================================================== main
def main():
    out = utils.results_dir('bonus')
    size = (640, 480)

    # 1. kappa of every synthetic flight, from our own matches
    frames, F, kappas, rows = {}, {}, {}, []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for fl in ('corniche', 'loop', 'lost_at_sea'):
        print(f'... estimating kappa on {fl}', flush=True)
        frames[fl] = utils.load_frames(utils.flight(fl))
        F[fl] = [features(im) for im in frames[fl]]
        k, (grid, coarse, c0, c1) = estimate_kappa(F[fl], size)
        kappas[fl] = k
        rows.append([fl, f'{k:.3f}', c0, c1])
        ax.plot(grid, coarse, '-o', ms=3, label=f'{fl}  (kappa = {k:.3f})')
    ax.set_xlabel('kappa'); ax.set_ylabel('mean homography residual (px)'); ax.grid(alpha=.3); ax.legend()
    ax.set_title('Homography fit against kappa')
    fig.tight_layout(); fig.savefig(f'{out}/kappa_curves.png', dpi=130); plt.close(fig)
    print('\nkappa from our own matches')
    utils.table(f'{out}/kappa.md', ['flight', 'kappa', 'residual at kappa = 0 (px)', 'residual at kappa (px)'],
                rows)

    # 2. one loop frame before (left) and after (right) undistorting
    img = utils.load_frames(utils.flight('loop'), color=True)[20]
    und, _ = undistort_image(img, kappas['loop'])
    both = np.hstack([img, np.full((img.shape[0], 10, 3), 255, np.uint8), und])
    cv2.imwrite(f'{out}/undistort_loop.jpg', both)

    # 3. Q3.2 again, on frames undistorted with each flight's own kappa
    rows, Th = [], {}
    for fl in ('corniche', 'loop'):
        print(f'... chaining {fl}, before and after undistorting', flush=True)
        Fu = [features_undistorted(im, kappas[fl]) for im in frames[fl]]
        runs = {}
        for tag, FF in (('before', F[fl]), ('after', Fu)):
            for model in ('similarity', 'homography'):
                T, _, lost, _ = chain(FF, model)
                e = gt.path_errors(utils.flight(fl), T)
                rows.append([fl, tag, model, len(lost), float(e.mean()), float(e[-1])])
                if tag == 'after':
                    runs[model] = T
        if fl == 'loop':
            viz.plot_paths(utils.flight(fl), runs, f'{out}/paths_loop.png')
        F[fl + '_u'] = Fu
    print('\nQ3.2 before and after undistorting')
    utils.table(f'{out}/models.md', ['flight', 'frames', 'model', 'lost steps', 'mean error (m)',
                                     'final error (m)'], rows)

    # 4. the match-error plot of Q3.2, on the undistorted frames
    pairs = []
    for fl, i, j in (('corniche', 20, 21), ('loop', 20, 21), ('loop', 3, 132)):
        (k1, d1), (k2, d2) = F[fl + '_u'][i], F[fl + '_u'][j]
        mm = match(d1, d2)
        pairs.append((f'{fl}, frames {i} and {j}, undistorted', k1[mm[:, 0]], k2[mm[:, 1]],
                      gt.gt_H(utils.flight(fl), i, j)))
    viz.plot_error_vs_radius(pairs, f'{out}/error_vs_radius.png')

    # 5. how much the answer depends on kappa: chain loop with kappas around ours
    rows = []
    for k in sorted({round(kappas['loop'] + d, 3) for d in (-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03)}):
        print(f'... loop with kappa = {k:.3f}', flush=True)
        Fk = [features_undistorted(im, k) for im in frames['loop']]
        e = {m: gt.path_errors(utils.flight('loop'), chain(Fk, m)[0]).mean() for m in ('similarity', 'homography')}
        rows.append([f'{k:.3f}', float(e['similarity']), float(e['homography'])])
    print('\nloop: mean error against the kappa used to undistort')
    utils.table(f'{out}/sensitivity_loop.md', ['kappa', 'similarity mean error (m)', 'homography mean error (m)'],
                rows)
    print(f'\nEverything is in {out}')


if __name__ == '__main__':
    utils.run(main)