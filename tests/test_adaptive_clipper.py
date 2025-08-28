import numpy as np
import dp_utils


def run_sim(dist_fn, rounds=120, clip=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    clipper = dp_utils.AdaptiveClipper(clip)
    for t in range(rounds):
        norms = dist_fn(rng, t)
        q90 = float(np.percentile(norms, 90))
        clipped_fraction = float(np.mean(norms > clipper.clip))
        clipper.update(clipped_fraction, q90)
    return clipper, clipped_fraction


def test_stationary_lognormal():
    def dist(rng, _):
        return rng.lognormal(mean=0.0, sigma=1.0, size=1000)
    clipper, clipped_fraction = run_sim(dist)
    assert abs(clipped_fraction - clipper.target) <= 0.03


def test_drift_recovery():
    def dist(rng, t):
        sigma = 1.0 if t < 60 else 1.5
        return rng.lognormal(mean=0.0, sigma=sigma, size=1000)
    clipper = dp_utils.AdaptiveClipper(1.0)
    rng = np.random.default_rng(1)
    recovered = False
    overshoot = 0.0
    for t in range(120):
        norms = dist(rng, t)
        q90 = float(np.percentile(norms, 90))
        clipped_fraction = float(np.mean(norms > clipper.clip))
        clipper.update(clipped_fraction, q90)
        if t >= 60 and not recovered:
            if abs(clipped_fraction - clipper.target) <= 0.05:
                recovered = True
        if t >= 60:
            overshoot = max(overshoot, abs(clipped_fraction - clipper.target))
    assert recovered
    assert overshoot <= 0.20


def test_heavy_tail_robustness():
    def dist(rng, _):
        base = rng.lognormal(mean=0.0, sigma=1.0, size=1000)
        n_out = max(1, int(0.01 * base.size))
        base[:n_out] = rng.pareto(2.0, size=n_out) * 20
        return base
    clipper, clipped_fraction = run_sim(dist, rng=np.random.default_rng(2))
    assert abs(clipped_fraction - clipper.target) <= 0.05
