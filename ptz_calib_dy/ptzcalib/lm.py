"""Levenberg-Marquardt trust-region solver mirroring Ceres 1.14.

The public C++ code drives every optimisation through ``ceres::Solve`` with the
default (Levenberg-Marquardt) trust-region strategy and numeric ``CENTRAL``
differentiation.  This module reproduces that machinery so that the Python port
follows the same accept/reject/termination decisions:

* ``TrustRegionMinimizer::Minimize``   -> :func:`minimize`
* ``LevenbergMarquardtStrategy``       -> radius / diagonal handling in the loop
* ``NumericDiffCostFunction<CENTRAL>`` -> :func:`central_difference_step`
* ``DENSE_QR``                         -> :class:`DenseNumericSystem`
* ``SPARSE_SCHUR``                     -> :class:`SchurSystem`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

MIN_NUMERIC_DIFF_STEP = math.sqrt(np.finfo(np.float64).eps)
RELATIVE_STEP_SIZE = 1e-6


def central_difference_step(x: np.ndarray) -> np.ndarray:
    """``std::max(min_step_size, |x| * relative_step_size)`` (numeric_diff.h)."""
    return np.maximum(MIN_NUMERIC_DIFF_STEP, np.abs(x) * RELATIVE_STEP_SIZE)


@dataclass
class SolverOptions:
    """Subset of ``ceres::Solver::Options`` that the port relies on."""

    max_num_iterations: int = 100
    function_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-8
    initial_trust_region_radius: float = 1e4
    max_trust_region_radius: float = 1e16
    min_trust_region_radius: float = 1e-32
    min_relative_decrease: float = 1e-3
    min_lm_diagonal: float = 1e-6
    max_lm_diagonal: float = 1e32
    max_num_consecutive_invalid_steps: int = 5
    jacobi_scaling: bool = True
    minimizer_progress_to_stdout: bool = False


# ``ceres::TerminationType``
CONVERGENCE = "CONVERGENCE"
NO_CONVERGENCE = "NO_CONVERGENCE"
FAILURE = "FAILURE"


@dataclass
class SolverSummary:
    """Subset of ``ceres::Solver::Summary``."""

    termination_type: str = NO_CONVERGENCE
    message: str = ""
    initial_cost: float = 0.0
    final_cost: float = 0.0
    num_residuals: int = 0
    num_parameters: int = 0
    num_successful_steps: int = 0
    num_unsuccessful_steps: int = 0
    iterations: int = 0

    def brief_report(self) -> str:
        return (
            f"Ceres Solver Report: Iterations: {self.iterations}, "
            f"Initial cost: {self.initial_cost:.6e}, Final cost: {self.final_cost:.6e}, "
            f"Termination: {self.termination_type}"
        )


class TrustRegionSystem:
    """Interface expected by :func:`minimize`."""

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def linearize(self, x: np.ndarray, f: np.ndarray) -> None:
        raise NotImplementedError

    def col_sq_norms(self) -> np.ndarray:
        raise NotImplementedError

    def gradient(self) -> np.ndarray:
        raise NotImplementedError

    def solve_step(self, scaling: np.ndarray, lm_diagonal: np.ndarray) -> Optional[np.ndarray]:
        """Solve ``(S J' J S + diag(D)) y = -S J' f``; returns ``y`` (scaled space)."""
        raise NotImplementedError

    def model_terms(self, delta: np.ndarray) -> Tuple[float, float]:
        """Returns ``((J delta).f, (J delta).(J delta))``."""
        raise NotImplementedError


def minimize(system: TrustRegionSystem, x0: np.ndarray, options: SolverOptions) -> Tuple[np.ndarray, SolverSummary]:
    """Port of ``ceres::internal::TrustRegionMinimizer::Minimize``."""
    summary = SolverSummary()

    x = np.array(x0, dtype=np.float64, copy=True)
    summary.num_parameters = int(x.size)

    # --- IterationZero -----------------------------------------------------
    f = system.evaluate(x)
    cost = 0.5 * float(f @ f)
    summary.initial_cost = cost
    summary.num_residuals = int(f.size)

    system.linearize(x, f)
    if options.jacobi_scaling:
        scaling = 1.0 / (1.0 + np.sqrt(system.col_sq_norms()))
    else:
        scaling = np.ones(x.size, dtype=np.float64)

    g = system.gradient()
    gradient_max_norm = float(np.max(np.abs(g))) if g.size else 0.0

    x_norm = -1.0  # Init() leaves x_norm_ invalid until the first successful step
    radius = options.initial_trust_region_radius
    decrease_factor = 2.0
    reuse_diagonal = False
    diagonal = np.zeros(x.size, dtype=np.float64)
    num_consecutive_invalid_steps = 0

    step_is_successful = True  # iteration zero counts as successful
    iteration = 0

    while True:
        # --- FinalizeIterationAndCheckIfMinimizerCanContinue ---------------
        if step_is_successful:
            summary.num_successful_steps += 1
        else:
            summary.num_unsuccessful_steps += 1

        if iteration >= options.max_num_iterations:
            summary.termination_type = NO_CONVERGENCE
            summary.message = "Maximum number of iterations reached."
            break

        if step_is_successful and gradient_max_norm <= options.gradient_tolerance:
            summary.termination_type = CONVERGENCE
            summary.message = f"Gradient tolerance reached. Gradient max norm: {gradient_max_norm:e}"
            break

        if radius <= options.min_trust_region_radius:
            summary.termination_type = CONVERGENCE
            summary.message = f"Minimum trust region radius reached. Trust region radius: {radius:e}"
            break

        iteration += 1

        # --- ComputeTrustRegionStep ---------------------------------------
        if not reuse_diagonal:
            diagonal = np.clip(
                system.col_sq_norms() * scaling * scaling,
                options.min_lm_diagonal,
                options.max_lm_diagonal,
            )
        lm_diagonal = diagonal / radius

        step_is_valid = False
        model_cost_change = 0.0
        delta = None

        y = system.solve_step(scaling, lm_diagonal)
        if y is not None and np.all(np.isfinite(y)):
            delta = scaling * y
            jd_f, jd_jd = system.model_terms(delta)
            model_cost_change = -(jd_f + 0.5 * jd_jd)
            step_is_valid = model_cost_change > 0.0

        if not step_is_valid:
            # HandleInvalidStep
            num_consecutive_invalid_steps += 1
            if num_consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps:
                summary.termination_type = FAILURE
                summary.message = (
                    "Number of consecutive invalid steps more than "
                    f"Solver::Options::max_num_consecutive_invalid_steps: "
                    f"{options.max_num_consecutive_invalid_steps}"
                )
                break
            # LevenbergMarquardtStrategy::StepIsInvalid -> StepRejected(0.0)
            radius = radius / decrease_factor
            decrease_factor *= 2.0
            reuse_diagonal = True
            step_is_successful = False
            continue

        num_consecutive_invalid_steps = 0

        # --- ComputeCandidatePointAndEvaluateCost -------------------------
        candidate_x = x + delta
        f_candidate = system.evaluate(candidate_x)
        if not np.all(np.isfinite(f_candidate)):
            candidate_cost = np.finfo(np.float64).max
        else:
            candidate_cost = 0.5 * float(f_candidate @ f_candidate)

        # --- ParameterToleranceReached ------------------------------------
        step_norm = float(np.linalg.norm(delta))
        step_size_tolerance = options.parameter_tolerance * (x_norm + options.parameter_tolerance)
        if step_norm <= step_size_tolerance:
            summary.termination_type = CONVERGENCE
            summary.message = "Parameter tolerance reached."
            break

        # --- FunctionToleranceReached -------------------------------------
        cost_change = cost - candidate_cost
        if abs(cost_change) <= options.function_tolerance * cost:
            summary.termination_type = CONVERGENCE
            summary.message = "Function tolerance reached."
            break

        # --- IsStepSuccessful ---------------------------------------------
        relative_decrease = cost_change / model_cost_change
        if relative_decrease > options.min_relative_decrease:
            # HandleSuccessfulStep
            x = candidate_x
            x_norm = float(np.linalg.norm(x))
            f = f_candidate
            cost = candidate_cost

            system.linearize(x, f)
            g = system.gradient()
            gradient_max_norm = float(np.max(np.abs(g))) if g.size else 0.0

            step_is_successful = True
            # LevenbergMarquardtStrategy::StepAccepted
            radius = radius / max(1.0 / 3.0, 1.0 - (2.0 * relative_decrease - 1.0) ** 3)
            radius = min(options.max_trust_region_radius, radius)
            decrease_factor = 2.0
            reuse_diagonal = False
        else:
            # HandleUnsuccessfulStep
            step_is_successful = False
            radius = radius / decrease_factor
            decrease_factor *= 2.0
            reuse_diagonal = True

        if options.minimizer_progress_to_stdout:
            print(
                f"{iteration:4d} cost: {cost:.6e} rd: {relative_decrease:+.2e} "
                f"radius: {radius:.2e} |g|: {gradient_max_norm:.2e}"
            )

    summary.final_cost = cost
    summary.iterations = iteration
    return x, summary


# --------------------------------------------------------------------------
# Dense system (ceres DENSE_QR)
# --------------------------------------------------------------------------
class DenseNumericSystem(TrustRegionSystem):
    """Explicit Jacobian, computed by central differences, solved via QR.

    ``residual_fn`` must accept an ``(n_eval, n)`` batch of parameter vectors and
    return an ``(n_eval, m)`` batch of residuals so that the finite differences
    stay vectorised.
    """

    def __init__(self, residual_fn: Callable[[np.ndarray], np.ndarray]):
        self._residual_fn = residual_fn
        self._J: Optional[np.ndarray] = None
        self._f: Optional[np.ndarray] = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._residual_fn(x[None, :])[0]

    def linearize(self, x: np.ndarray, f: np.ndarray) -> None:
        n = x.size
        delta = central_difference_step(x)
        xs = np.repeat(x[None, :], 2 * n, axis=0)
        idx = np.arange(n)
        xs[idx, idx] += delta
        xs[n + idx, idx] -= delta
        res = self._residual_fn(xs)  # (2n, m)
        self._J = ((res[:n] - res[n:]) / (2.0 * delta)[:, None]).T  # (m, n)
        self._f = f

    def col_sq_norms(self) -> np.ndarray:
        return np.einsum("ij,ij->j", self._J, self._J)

    def gradient(self) -> np.ndarray:
        return self._J.T @ self._f

    def solve_step(self, scaling: np.ndarray, lm_diagonal: np.ndarray) -> Optional[np.ndarray]:
        A = self._J * scaling[None, :]
        n = A.shape[1]
        aug = np.vstack([A, np.diag(np.sqrt(lm_diagonal))])
        rhs = np.concatenate([self._f, np.zeros(n)])
        try:
            y, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return -y  # ceres solves Jy = r then negates

    def model_terms(self, delta: np.ndarray) -> Tuple[float, float]:
        jd = self._J @ delta
        return float(jd @ self._f), float(jd @ jd)


# --------------------------------------------------------------------------
# Schur-complement system (ceres SPARSE_SCHUR)
# --------------------------------------------------------------------------
class SchurSystem(TrustRegionSystem):
    """Bundle-adjustment style system with eliminable 3-dof "ray" blocks.

    Parameters are laid out as ``[camera params (n_cam), ray params (3 * n_ray)]``.
    The Jacobian is supplied in block form by :meth:`linearize` implementations
    of subclasses through :meth:`set_blocks`:

    ``cols_a`` (Na, ba)  camera columns touched by residual pair ``a``
    ``jac_a``  (Na, 2, ba)
    ``ray_idx``(Na,)     ray block touched by residual pair ``a``
    ``jac_r``  (Na, 2, 3)
    ``cols_b`` (Nb, bb)  camera columns of the ray-free residual pairs
    ``jac_b``  (Nb, 2, bb)
    """

    def __init__(self, n_cam: int, n_ray: int):
        self.n_cam = int(n_cam)
        self.n_ray = int(n_ray)
        self._cols_a = np.zeros((0, 0), dtype=np.int64)
        self._jac_a = np.zeros((0, 2, 0))
        self._ray_idx = np.zeros((0,), dtype=np.int64)
        self._jac_r = np.zeros((0, 2, 3))
        self._cols_b = np.zeros((0, 0), dtype=np.int64)
        self._jac_b = np.zeros((0, 2, 0))
        self._fa = np.zeros((0, 2))
        self._fb = np.zeros((0, 2))
        self._lin_a: Optional[np.ndarray] = None
        self._lin_b: Optional[np.ndarray] = None
        self._lin_e: Optional[np.ndarray] = None

    # -- to be filled in by linearize() ------------------------------------
    def set_blocks(self, cols_a, jac_a, ray_idx, jac_r, cols_b, jac_b, f) -> None:
        self._cols_a = cols_a
        self._jac_a = jac_a
        self._ray_idx = ray_idx
        self._jac_r = jac_r
        self._cols_b = cols_b
        self._jac_b = jac_b

        na = cols_a.shape[0]
        self._fa = f[: 2 * na].reshape(na, 2)
        self._fb = f[2 * na:].reshape(-1, 2)

        n_cam = self.n_cam
        # cached linear indices for the bincount-based accumulation
        self._lin_a = (cols_a[:, :, None] * n_cam + cols_a[:, None, :]).ravel()
        self._lin_b = (
            (cols_b[:, :, None] * n_cam + cols_b[:, None, :]).ravel()
            if cols_b.shape[0]
            else np.zeros(0, dtype=np.int64)
        )
        if na:
            self._lin_e = (
                (cols_a[:, :, None] * self.n_ray + ray_idx[:, None, None]) * 3
                + np.arange(3)[None, None, :]
            ).ravel()
        else:
            self._lin_e = np.zeros(0, dtype=np.int64)

    # -- TrustRegionSystem -------------------------------------------------
    def col_sq_norms(self) -> np.ndarray:
        out = np.zeros(self.n_cam + 3 * self.n_ray, dtype=np.float64)
        if self._cols_a.shape[0]:
            va = np.einsum("nij,nij->nj", self._jac_a, self._jac_a)
            out[: self.n_cam] += np.bincount(
                self._cols_a.ravel(), weights=va.ravel(), minlength=self.n_cam
            )
            vr = np.einsum("nij,nij->nj", self._jac_r, self._jac_r)
            lin_r = (self._ray_idx[:, None] * 3 + np.arange(3)[None, :]).ravel()
            out[self.n_cam:] += np.bincount(lin_r, weights=vr.ravel(), minlength=3 * self.n_ray)
        if self._cols_b.shape[0]:
            vb = np.einsum("nij,nij->nj", self._jac_b, self._jac_b)
            out[: self.n_cam] += np.bincount(
                self._cols_b.ravel(), weights=vb.ravel(), minlength=self.n_cam
            )
        return out

    def _gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        gc = np.zeros(self.n_cam, dtype=np.float64)
        gr = np.zeros((self.n_ray, 3), dtype=np.float64)
        if self._cols_a.shape[0]:
            va = np.einsum("nij,ni->nj", self._jac_a, self._fa)
            gc += np.bincount(self._cols_a.ravel(), weights=va.ravel(), minlength=self.n_cam)
            vr = np.einsum("nij,ni->nj", self._jac_r, self._fa)
            lin_r = (self._ray_idx[:, None] * 3 + np.arange(3)[None, :]).ravel()
            gr += np.bincount(lin_r, weights=vr.ravel(), minlength=3 * self.n_ray).reshape(self.n_ray, 3)
        if self._cols_b.shape[0]:
            vb = np.einsum("nij,ni->nj", self._jac_b, self._fb)
            gc += np.bincount(self._cols_b.ravel(), weights=vb.ravel(), minlength=self.n_cam)
        return gc, gr

    def gradient(self) -> np.ndarray:
        gc, gr = self._gradients()
        return np.concatenate([gc, gr.ravel()])

    def _hessian_blocks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_cam, n_ray = self.n_cam, self.n_ray
        B = np.zeros(n_cam * n_cam, dtype=np.float64)
        C = np.zeros((n_ray, 3, 3), dtype=np.float64)
        E = np.zeros(n_cam * n_ray * 3, dtype=np.float64)

        if self._cols_a.shape[0]:
            ba = np.einsum("nij,nik->njk", self._jac_a, self._jac_a)
            B += np.bincount(self._lin_a, weights=ba.ravel(), minlength=n_cam * n_cam)
            cr = np.einsum("nij,nik->njk", self._jac_r, self._jac_r)
            for j in range(3):
                for k in range(3):
                    C[:, j, k] = np.bincount(
                        self._ray_idx, weights=cr[:, j, k], minlength=n_ray
                    )
            ea = np.einsum("nij,nik->njk", self._jac_a, self._jac_r)
            E += np.bincount(self._lin_e, weights=ea.ravel(), minlength=n_cam * n_ray * 3)
        if self._cols_b.shape[0]:
            bb = np.einsum("nij,nik->njk", self._jac_b, self._jac_b)
            B += np.bincount(self._lin_b, weights=bb.ravel(), minlength=n_cam * n_cam)

        return B.reshape(n_cam, n_cam), C, E.reshape(n_cam, n_ray, 3)

    def solve_step(self, scaling: np.ndarray, lm_diagonal: np.ndarray) -> Optional[np.ndarray]:
        n_cam, n_ray = self.n_cam, self.n_ray
        B, C, E = self._hessian_blocks()
        gc, gr = self._gradients()

        sc = scaling[:n_cam]
        sr = scaling[n_cam:].reshape(n_ray, 3)

        # column scaling
        B = B * sc[:, None] * sc[None, :]
        C = C * sr[:, :, None] * sr[:, None, :]
        E = E * sc[:, None, None] * sr[None, :, :]
        gc = gc * sc
        gr = gr * sr

        # LM damping
        B[np.diag_indices(n_cam)] += lm_diagonal[:n_cam]
        dr = lm_diagonal[n_cam:].reshape(n_ray, 3)
        C[:, 0, 0] += dr[:, 0]
        C[:, 1, 1] += dr[:, 1]
        C[:, 2, 2] += dr[:, 2]

        try:
            Cinv = np.linalg.inv(C) if n_ray else C
        except np.linalg.LinAlgError:
            return None

        if n_ray:
            W = np.einsum("ctk,tkl->ctl", E, Cinv)  # (n_cam, n_ray, 3)
            Wf = W.reshape(n_cam, 3 * n_ray)
            Ef = E.reshape(n_cam, 3 * n_ray)
            S = B - Wf @ Ef.T
            rhs = -gc + Wf @ gr.ravel()
        else:
            S = B
            rhs = -gc

        try:
            yc = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            yc, *_ = np.linalg.lstsq(S, rhs, rcond=None)
        if not np.all(np.isfinite(yc)):
            return None

        if n_ray:
            etyc = np.einsum("ctk,c->tk", E, yc)
            yr = np.einsum("tkl,tl->tk", Cinv, -gr - etyc)
            return np.concatenate([yc, yr.ravel()])
        return yc

    def model_terms(self, delta: np.ndarray) -> Tuple[float, float]:
        n_cam = self.n_cam
        dc = delta[:n_cam]
        dr = delta[n_cam:].reshape(self.n_ray, 3)

        jd_f = 0.0
        jd_jd = 0.0
        if self._cols_a.shape[0]:
            jd = np.einsum("nij,nj->ni", self._jac_a, dc[self._cols_a])
            jd += np.einsum("nij,nj->ni", self._jac_r, dr[self._ray_idx])
            jd_f += float(np.einsum("ni,ni->", jd, self._fa))
            jd_jd += float(np.einsum("ni,ni->", jd, jd))
        if self._cols_b.shape[0]:
            jdb = np.einsum("nij,nj->ni", self._jac_b, dc[self._cols_b])
            jd_f += float(np.einsum("ni,ni->", jdb, self._fb))
            jd_jd += float(np.einsum("ni,ni->", jdb, jdb))
        return jd_f, jd_jd
