"""Cross-check the grouped central differences against a brute-force Jacobian."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from ptzcalib import data_io, lm
from ptzcalib.logging_utils import init_logging
from ptzcalib.ptzray_optimizer import PTZRayOptimizer, FactorType
from ptzcalib.types import Camera
import logging

init_logging(False); logging.getLogger('ptzcalib').setLevel(logging.WARNING)
cv2.setRNGSeed(0)

ROOT = '/home/dyyu/ptz-calib-python/PTZ-Calib/data/synthetic'
ok, fnames, feats, sizes = data_io.load_imgs_and_features(f'{ROOT}/offline/scene_01', f'{ROOT}/offline_matches/scene_01')
ok, mi = data_io.load_matches_info(f'{ROOT}/offline_matches/scene_01/pairs_matches.txt', fnames, feats)
ok, pixels, pts3d = data_io.load_annotation(f'{ROOT}/offline/scene_01/scene_01.json', fnames)

n = len(fnames)
cams = [Camera() for _ in range(n)]
for c, s in zip(cams, sizes):
    f = 1.2 * max(s)
    c.K = np.array([[f,0,0.5*s[0]],[0,f,0.5*s[1]],[0,0,1.0]])

def dense_jac(opt, x):
    m = opt.evaluate(x).size
    J = np.zeros((m, x.size))
    for i in range(x.size):
        d = lm.central_difference_step(x[i:i+1])[0]
        xp = x.copy(); xp[i] += d
        xm = x.copy(); xm[i] -= d
        J[:, i] = (opt.evaluate(xp) - opt.evaluate(xm)) / (2*d)
    return J

def block_to_dense(opt):
    m = 2*opt._num_obs_a + 2*opt._num_obs_b
    J = np.zeros((m, opt.n_cam + 3*opt.n_ray))
    for k in range(opt._num_obs_a):
        for r in range(2):
            J[2*k+r, opt._cam_cols_a[k]] += opt._jac_a[k, r]
            J[2*k+r, opt.n_cam + 3*opt._obs_ray[k] : opt.n_cam + 3*opt._obs_ray[k]+3] += opt._jac_r[k, r]
    off = 2*opt._num_obs_a
    for k in range(opt._num_obs_b):
        for r in range(2):
            J[off+2*k+r, opt._cam_cols_b[k]] += opt._jac_b[k, r]
    return J

for label, cam_ids, use_annot, ftype in [
    ("init-pair (PTZRay, 2 cams)", {0, 5}, False, FactorType.PTZRay),
    ("georef (PTZRay + 2d3d)", set(range(30)), True, FactorType.PTZRay),
    ("georef dist (PTZRayDist + 2d3d)", set(range(30)), True, FactorType.PTZRayDist),
]:
    kw = dict(pixels=pixels, pts3d=pts3d) if use_annot else {}
    opt = PTZRayOptimizer(feats, mi, cams, cam_ids, 200, ftype, **kw)
    opt._find_tracks(); opt._set_init_trans_local_to_world(); opt._set_up_initial_camera_params(); opt._build_problem()
    lm.SchurSystem.__init__(opt, opt.n_cam, opt.n_ray)
    x = opt._pack_params()
    f = opt.evaluate(x)
    t0 = time.time(); opt.linearize(x, f); t_grp = time.time()-t0
    Jb = block_to_dense(opt)
    # brute force on a random subset of columns (full is too slow)
    rng = np.random.default_rng(0)
    cols = np.concatenate([np.arange(min(opt.n_cam, x.size)), opt.n_cam + rng.choice(3*opt.n_ray, size=min(60, 3*opt.n_ray), replace=False)])
    err = 0.0; scale = 0.0
    for i in cols:
        d = lm.central_difference_step(x[i:i+1])[0]
        xp = x.copy(); xp[i] += d
        xm = x.copy(); xm[i] -= d
        col = (opt.evaluate(xp) - opt.evaluate(xm)) / (2*d)
        err = max(err, np.max(np.abs(col - Jb[:, i])))
        scale = max(scale, np.max(np.abs(col)))
    print(f"{label:36s} obs_a={opt._num_obs_a:6d} obs_b={opt._num_obs_b:4d} "
          f"n_cam={opt.n_cam:4d} n_ray={opt.n_ray:5d} | jac {t_grp*1000:6.1f} ms | max|dJ|={err:.3e} (scale {scale:.3e})")
