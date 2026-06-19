"""
Closed-form theoretical quantities from Section IV-B of the theory paper.

These are pure math functions, no model/data needed — given r_perp and
kappa_max (aggregated empirically elsewhere), everything here is exact
arithmetic from Eq. (6), (7), and (12).

Example:
    alpha_star = compute_alpha_star(r_perp=0.42, kappa_max=0.18, k=10)
    Ck = compute_Ck(k=10)                      # -> 3.4868 (matches paper's "≈3.49")
    d  = worst_case_deficit(alpha=0.504, r_perp=0.42, kappa_max=0.18, k=10)
"""

import math


def compute_alpha_star(r_perp: float, kappa_max: float, k: int) -> float:
    """
    Eq. (7): alpha* = (k-1)*kappa_max / (r_perp + (k-1)*kappa_max)

    This is the threshold above which F^sub_theta is monotone submodular
    (Theorem A / Theorem 1).
    """
    a = (k - 1) * kappa_max
    b = r_perp
    if a + b <= 0:
        return 0.0
    return a / (a + b)


def worst_case_deficit(alpha: float, r_perp: float, kappa_max: float, k: int) -> float:
    """
    Eq. (6): delta(alpha) <= max(0, (1-alpha)*(k-1)*kappa_max - alpha*r_perp)

    This is the WORST-CASE bound on the monotonicity deficit (Definition 3),
    not the empirically-estimated average deficit (Definition 4) — for that,
    see DiversityModule.estimate_average_deficit in diversity.py.
    """
    a = (k - 1) * kappa_max
    b = r_perp
    return max(0.0, (1 - alpha) * a - alpha * b)


def compute_Ck(k: int) -> float:
    """
    Eq. (12): closed-form constant in the near-monotone bound (Theorem 2).

        Ck = (k-1)(1 - rho^k) - k*rho*(1 - k*rho^{k-1} + (k-1)*rho^k)
        rho = 1 - 1/k

    The paper's PDF garbles the bracketed term during text extraction
    ("ρk1 − kρk−1 + (k−1)ρk" should read "rho*k*[1 - k*rho^(k-1) +
    (k-1)*rho^k]"). I reconstructed the exact form from the telescoping-sum
    proof leading to Eq. (16) (using the standard closed form for
    sum_j j*rho^j) and verified numerically: compute_Ck(10) = 3.4868,
    matching the paper's reported "C10 ≈ 3.49".

    Asymptotically Ck -> (k-1)/e as k grows (stated in the paper); this
    closed form is exact for any finite k, not just the limit.
    """
    if k <= 1:
        return 0.0
    rho = 1 - 1.0 / k
    term1 = (k - 1) * (1 - rho ** k)
    term2 = k * rho * (1 - k * rho ** (k - 1) + (k - 1) * rho ** k)
    return term1 - term2


def asymptotic_Ck(k: int) -> float:
    """The paper's stated asymptotic approximation Ck ~ (k-1)/e, for comparison."""
    return (k - 1) / math.e


def near_monotone_bound(opt: float, alpha: float, r_perp: float,
                        kappa_max: float, k: int) -> float:
    """
    Theorem 2: F(Sg) >= (1 - 1/e)*OPT - Ck*delta(alpha)
    Returns the right-hand side (the guaranteed lower bound on F(Sg)).
    """
    Ck = compute_Ck(k)
    delta = worst_case_deficit(alpha, r_perp, kappa_max, k)
    return (1 - 1 / math.e) * opt - Ck * delta


def expected_deficit_upper_bound(alpha_mean: float, alpha_var: float,
                                alpha_star: float, r_perp: float,
                                kappa_max: float, k: int) -> float:
    """
    Proposition 2, Eq. (19)-(20):
        E[delta(alpha_t)] <= (r_perp + (k-1)*kappa_max) * sqrt(Var(alpha_t) + (alpha* - alpha_mean)^2)

    This decomposes the expected gap into "distance of the policy's
    (mean, std) from the ideal point (alpha*, 0)".
    """
    coeff = r_perp + (k - 1) * kappa_max
    dist = math.sqrt(alpha_var + (alpha_star - alpha_mean) ** 2)
    return coeff * dist