#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster -- entry for make.fun's
gta6-trophy-rarity-forecast-01.

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
<id>.out forecast per case to $MAKEFUN_OUTPUT_DIR (else /output): for
each trophy, five quantiles (levels 0.05/0.25/0.5/0.75/0.95) of its
Exophase earn rate, in percent, at the case's target instant.

Model, built from the competition rules:

  * Exophase refreshes its counts in jumps, so a case's read history is
    first collapsed to refresh points -- the reads where the tracked
    player count or any trophy's rate actually changed.  Everything
    else is the same frozen page read again.
  * Players at the target come from an ensemble of two fits over the
    refresh points: a saturating exponential (the tracked pool levels
    off) and a log-linear fit of ln(players) against ln(1 + hours).
    The two are combined by geometric mean and never project below the
    last observed player count.
  * Earners per trophy (rate x players) are projected with an
    elasticity model: how fast earners grow as the player pool grows,
    estimated per tier pool (platinum/gold/silver/bronze) from the
    log-log deltas between refresh points, each pool shrunk toward the
    global elasticity by its sample size, each trophy shrunk toward
    its pool.  A trophy whose earners lag the pool's growth dilutes as
    new players flood in; one that keeps pace holds its rate.
  * The spread is per trophy: the RMS residual of its logit rate
    around its own drift over the refresh points (a pooled value when
    a trophy has too few points), scaled by sqrt(horizon in days),
    floored and capped, and widened by half again when the trophy's
    rate jumped sharply at the last refresh.
  * A history that is a single frozen page falls back to persistence:
    the median is the last rate and the spread is the house
    random-walk scale, widened because a refresh jump is likely
    sitting unseen between the checkpoint and the target.

Python 3 standard library only, deterministic, no network.
"""

import glob
import json
import math
import os
from datetime import datetime, timezone

LEVELS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

RATE_LO = 0.01   # the scorer clamps rates into [0.01, 99.99] on the logit scale
RATE_HI = 99.99

SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.5
SHARP_LOGIT = 1.0        # a last-refresh logit move this big is a regime jump
SHARP_WIDEN = 1.5
EARN_FLOOR = 0.05        # players; keeps log(earners) finite at a 0.00% rate
POOL_PRIOR = 10.0        # shrinkage strength, in (interval, trophy) pairs
TROPHY_PRIOR = 8.0
HALF_LIFE_DAYS = 1.0     # recency weighting for intervals and residuals
PLAYERS_CAP_MULT = 200.0 # sanity cap on the player projection


def clamp_rate(x):
    if x < RATE_LO:
        return RATE_LO
    if x > RATE_HI:
        return RATE_HI
    return x


def logit_pct(x):
    p = clamp_rate(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0.0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def z_for_level(level):
    key = round(float(level), 2)
    if key in Z_TABLE:
        return Z_TABLE[key]
    return _norm_ppf(float(level))


def _norm_ppf(p):
    """Inverse normal CDF (Acklam's rational approximation).

    Only used if a case ever names a level outside the five standard
    ones in Z_TABLE.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds, or None."""
    if s is None:
        return None
    txt = str(s).strip()
    if txt.endswith('Z'):
        txt = txt[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def recency_weight(t, t_ref):
    age_days = (t_ref - t) / 86400.0
    if age_days <= 0.0:
        return 1.0
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def _wmean(xs, ws):
    sw = sum(ws)
    if sw <= 0.0:
        return 0.0
    return sum(w * x for x, w in zip(xs, ws)) / sw


def _wslope_intercept(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; (a, b) or None."""
    sw = sum(ws)
    if sw <= 0.0:
        return None
    mx = _wmean(xs, ws)
    my = _wmean(ys, ws)
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


def _rates_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None or y is None:
            if x is not y:
                return False
        elif x != y:
            return False
    return True


def collapse_refresh(history):
    """Collapse a read history to refresh points: the first read, plus
    every read whose player count or any rate differs from the last
    kept read.  Returns [{'t', 'players', 'rates'}, ...]."""
    pts = []
    for read in history or []:
        if not isinstance(read, dict):
            continue
        players = read.get('players_tracked')
        t = parse_time(read.get('at') or read.get('slot'))
        if players is None or t is None:
            continue
        rates = read.get('rates')
        if not isinstance(rates, list):
            rates = []
        if pts and pts[-1]['players'] == players and \
                _rates_equal(pts[-1]['rates'], rates):
            continue
        pts.append({'t': t, 'players': float(players), 'rates': list(rates)})
    return pts


def last_rate(pts, idx):
    """The trophy's most recent non-null rate across the points."""
    for p in reversed(pts):
        if idx < len(p['rates']):
            r = p['rates'][idx]
            if r is not None:
                return float(r)
    return None


def _fit_saturating(xs, ps, tT):
    """Least-squares P(t) = Pinf - (Pinf - P0) * exp(-k t) by a
    deterministic grid search over (Pinf, k) plus one refinement pass.
    xs are hours since the first point, ps the player counts."""
    p0 = ps[0]
    pmax = max(ps)
    if pmax <= p0:
        return pmax

    def sse(pinf, k):
        amp = pinf - p0
        tot = 0.0
        for x, p in zip(xs, ps):
            pred = pinf - amp * math.exp(-k * x)
            tot += (pred - p) ** 2
        return tot

    factors = (1.01, 1.03, 1.06, 1.10, 1.15, 1.25, 1.40, 1.60, 1.85, 2.15,
               2.50, 3.0, 3.6, 4.3, 5.2, 6.5, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0)
    ks = [10.0 ** (-4.0 + 4.3 * i / 43.0) for i in range(44)]
    best = None
    for f in factors:
        pinf = pmax * f
        for k in ks:
            v = sse(pinf, k)
            if best is None or v < best[0]:
                best = (v, pinf, k)
    _, pinf0, k0 = best
    for fs in (0.8, 0.9, 1.0, 1.1, 1.2):
        pinf = pmax + (pinf0 - pmax) * fs
        if pinf <= pmax:
            continue
        for fk in (0.85, 0.925, 1.0, 1.075, 1.15):
            v = sse(pinf, k0 * fk)
            if v < best[0]:
                best = (v, pinf, k0 * fk)
    _, pinf, k = best
    return pinf - (pinf - p0) * math.exp(-k * tT)


def _fit_loglinear(xs, ps, tT):
    """Least squares of ln P against ln(1 + hours); value at tT."""
    X = [math.log1p(x) for x in xs]
    Y = [math.log(p) for p in ps]
    fit = _wslope_intercept(X, Y, [1.0] * len(X))
    if fit is None:
        return None
    a, b = fit
    v = a + b * math.log1p(tT)
    if v > 21.0:
        v = 21.0
    return math.exp(v)


def project_players(pts, t_target):
    """Ensemble player count at the target: geometric mean of the
    saturating-exponential and log-linear fits, at least the last
    observed count, with a generous sanity cap."""
    p_last = max(pts[-1]['players'], 1.0)
    t0 = pts[0]['t']
    xs = [(p['t'] - t0) / 3600.0 for p in pts]
    ps = [max(p['players'], 1.0) for p in pts]
    tT = max((t_target - t0) / 3600.0, xs[-1])
    preds = []
    for fit in (_fit_saturating, _fit_loglinear):
        try:
            v = fit(xs, ps, tT)
        except (OverflowError, ValueError):
            v = None
        if v is not None and math.isfinite(v) and v > 0.0:
            preds.append(v)
    if not preds:
        return p_last
    if len(preds) == 2:
        pred = math.sqrt(preds[0] * preds[1])
    else:
        pred = preds[0]
    if pred < p_last:
        pred = p_last
    cap = PLAYERS_CAP_MULT * p_last
    if pred > cap:
        pred = cap
    return pred


def _earners(rate, players):
    return rate / 100.0 * players


def estimate_betas(pts, trophies):
    """Per-trophy earners-vs-players elasticity beta.

    For every interval between consecutive refresh points with player
    growth, regress dln(earners) on dln(players) through the origin,
    recency-weighted.  Pool the intervals by trophy tier; shrink each
    pool toward the global beta by its pair count, and each trophy
    toward its pool.  Betas are clamped to [0, 2]: earners are a
    cumulative count, so a negative elasticity is measurement noise,
    and above 2 is extrapolation madness.
    """
    n = len(trophies)
    tiers = [t.get('tier') if isinstance(t, dict) else None for t in trophies]
    t_ref = pts[-1]['t']
    g_num = 0.0
    g_den = 0.0
    pool_num = {}
    pool_den = {}
    pool_cnt = {}
    per_num = [0.0] * n
    per_den = [0.0] * n
    per_cnt = [0] * n
    for j in range(1, len(pts)):
        p_prev = max(pts[j - 1]['players'], 1.0)
        p_cur = max(pts[j]['players'], 1.0)
        dlnp = math.log(p_cur) - math.log(p_prev)
        if dlnp <= 1e-9:
            continue
        w = recency_weight(pts[j]['t'], t_ref)
        r_prev = pts[j - 1]['rates']
        r_cur = pts[j]['rates']
        for i in range(n):
            if i >= len(r_prev) or i >= len(r_cur):
                continue
            a = r_prev[i]
            b = r_cur[i]
            if a is None or b is None:
                continue
            de = math.log(max(_earners(float(b), p_cur), EARN_FLOOR)) - \
                math.log(max(_earners(float(a), p_prev), EARN_FLOOR))
            g_num += w * dlnp * de
            g_den += w * dlnp * dlnp
            tier = tiers[i]
            pool_num[tier] = pool_num.get(tier, 0.0) + w * dlnp * de
            pool_den[tier] = pool_den.get(tier, 0.0) + w * dlnp * dlnp
            pool_cnt[tier] = pool_cnt.get(tier, 0) + 1
            per_num[i] += w * dlnp * de
            per_den[i] += w * dlnp * dlnp
            per_cnt[i] += 1
    beta_global = g_num / g_den if g_den > 1e-12 else None
    betas = []
    for i in range(n):
        tier = tiers[i]
        pool_beta = None
        if pool_den.get(tier, 0.0) > 1e-12:
            raw_pool = pool_num[tier] / pool_den[tier]
            if beta_global is not None:
                c = pool_cnt.get(tier, 0)
                pool_beta = (c * raw_pool + POOL_PRIOR * beta_global) / \
                            (c + POOL_PRIOR)
            else:
                pool_beta = raw_pool
        elif beta_global is not None:
            pool_beta = beta_global
        if pool_beta is None:
            pool_beta = 0.0
        if per_den[i] > 1e-12:
            raw_i = per_num[i] / per_den[i]
            beta_i = (per_cnt[i] * raw_i + TROPHY_PRIOR * pool_beta) / \
                     (per_cnt[i] + TROPHY_PRIOR)
        else:
            beta_i = pool_beta
        if beta_i < 0.0:
            beta_i = 0.0
        elif beta_i > 2.0:
            beta_i = 2.0
        betas.append(beta_i)
    return betas


def estimate_sigmas(pts, trophies, horizon_days):
    """Per-trophy logit-scale spread: RMS residual of the trophy's
    logit rate around its own recency-weighted linear drift over the
    refresh points (pooled across trophies when a trophy has fewer
    than 3 usable points), scaled by sqrt(horizon days), widened when
    the rate jumped at the last refresh, floored and capped."""
    n = len(trophies)
    t_ref = pts[-1]['t']
    t0 = pts[0]['t']
    scale = math.sqrt(horizon_days) if horizon_days > 0.0 else 0.0
    own = [None] * n
    pool_num = 0.0
    pool_den = 0.0
    for i in range(n):
        xs = []
        ys = []
        ws = []
        for p in pts:
            if i < len(p['rates']) and p['rates'][i] is not None:
                xs.append((p['t'] - t0) / 86400.0)
                ys.append(logit_pct(float(p['rates'][i])))
                ws.append(recency_weight(p['t'], t_ref))
        if len(xs) >= 3:
            fit = _wslope_intercept(xs, ys, ws)
            if fit is not None:
                a, b = fit
                num = 0.0
                den = 0.0
                for x, y, w in zip(xs, ys, ws):
                    r = y - (a + b * x)
                    num += w * r * r
                    den += w
                if den > 0.0:
                    own[i] = math.sqrt(num / den)
                    pool_num += num
                    pool_den += den
    pooled = math.sqrt(pool_num / pool_den) if pool_den > 0.0 else 0.5
    sigmas = []
    for i in range(n):
        base = own[i] if own[i] is not None else pooled
        s = base * scale
        if len(pts) >= 2:
            r_prev = pts[-2]['rates']
            r_last = pts[-1]['rates']
            if i < len(r_prev) and i < len(r_last) and \
                    r_prev[i] is not None and r_last[i] is not None:
                jump = abs(logit_pct(float(r_last[i])) -
                           logit_pct(float(r_prev[i])))
                if jump >= SHARP_LOGIT:
                    s *= SHARP_WIDEN
        if s < SIGMA_FLOOR:
            s = SIGMA_FLOOR
        elif s > SIGMA_CAP:
            s = SIGMA_CAP
        sigmas.append(s)
    return sigmas


def build_row(tid, median, sigma, levels):
    lm = logit_pct(median)
    out = []
    prev = None
    for lv in levels:
        q = clamp_rate(inv_logit_pct(lm + z_for_level(lv) * sigma))
        v = round(q, 4)
        if prev is not None and v < prev:
            v = prev
        out.append(v)
        prev = v
    return {'id': tid, 'q': out}


def _flat_row(tid, levels):
    return {'id': tid, 'q': [RATE_LO] * len(levels)}


def forecast_case(case):
    trophies = case['trophies']
    levels = case.get('levels') or LEVELS_DEFAULT
    horizon_min = case.get('horizon_minutes')
    t_at = parse_time(case.get('at'))
    t_target = parse_time(case.get('target'))
    if horizon_min is None and t_at is not None and t_target is not None:
        horizon_min = (t_target - t_at) / 60.0
    horizon_days = max(float(horizon_min or 0.0), 0.0) / 1440.0
    pts = collapse_refresh(case.get('history'))
    ids = [t.get('id') if isinstance(t, dict) else None for t in trophies]

    if len(pts) <= 1:
        # One frozen page (or no reads at all): persistence, with the
        # house random-walk spread widened for the unseen refresh.
        sigma = 0.5 * math.sqrt(horizon_days) * 1.5 if horizon_days > 0.0 else 0.0
        rows = []
        for i, tid in enumerate(ids):
            r = last_rate(pts, i)
            if r is None:
                rows.append(_flat_row(tid, levels))
            else:
                rows.append(build_row(tid, r, sigma, levels))
        return rows

    if t_target is None:
        t_target = pts[-1]['t'] + horizon_days * 86400.0
    p_target = project_players(pts, t_target)
    betas = estimate_betas(pts, trophies)
    sigmas = estimate_sigmas(pts, trophies, horizon_days)
    rows = []
    for i, tid in enumerate(ids):
        li = None
        for j in range(len(pts) - 1, -1, -1):
            if i < len(pts[j]['rates']) and pts[j]['rates'][i] is not None:
                li = j
                break
        if li is None:
            rows.append(_flat_row(tid, levels))
            continue
        p_at = max(pts[li]['players'], 1.0)
        e_last = _earners(float(pts[li]['rates'][i]), p_at)
        e_t = e_last * (p_target / p_at) ** betas[i]
        if e_t > p_target:
            e_t = p_target
        median = 100.0 * e_t / p_target
        rows.append(build_row(tid, median, sigmas[i], levels))
    return rows


def fallback_rows(case):
    """Emergency persistence forecast, used only if the main model
    ever fails on a parsed case: a valid file beats a missing one."""
    trophies = case['trophies']
    levels = case.get('levels') or LEVELS_DEFAULT
    horizon_days = max(float(case.get('horizon_minutes') or 0.0), 0.0) / 1440.0
    sigma = 0.5 * math.sqrt(horizon_days)
    pts = collapse_refresh(case.get('history'))
    rows = []
    for i, t in enumerate(trophies):
        tid = t.get('id') if isinstance(t, dict) else None
        r = last_rate(pts, i)
        if r is None:
            rows.append(_flat_row(tid, levels))
        else:
            rows.append(build_row(tid, r, sigma, levels))
    return rows


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, '*.in'))):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                case = json.load(fh)
        except Exception:
            continue
        cid = case.get('id') or os.path.splitext(os.path.basename(path))[0]
        try:
            rows = forecast_case(case)
        except Exception:
            try:
                rows = fallback_rows(case)
            except Exception:
                continue
        text = json.dumps({'trophies': rows}, separators=(',', ':'),
                          ensure_ascii=False)
        if len(text.encode('utf-8')) > 65536:
            # Over the size cap: shed decimals (rounding is monotone,
            # so the quantiles stay non-decreasing). 55 trophies never
            # get near this; it guards a case with far more trophies.
            rows = [{'id': r['id'], 'q': [round(v, 2) for v in r['q']]}
                    for r in rows]
            text = json.dumps({'trophies': rows}, separators=(',', ':'),
                              ensure_ascii=False)
        with open(os.path.join(out_dir, str(cid) + '.out'), 'w',
                  encoding='utf-8') as fh:
            fh.write(text)


if __name__ == '__main__':
    main()
