"""
EMAEvalHook — online evaluation-side EMA for the control experiment (tab:ema_control).

Replaces the offline ema_eval.py approach (per-round states were never saved).
Maintains the EMA buffer DURING training and evaluates both the raw and the
EMA-smoothed model each round, so raw-vs-EMA is internally paired within one run
and immune to rerun nondeterminism.

Integration into fl_server.py (~5 lines):

    from analysis.ema_eval_hook import EMAEvalHook
    hook = EMAEvalHook(beta=0.5, space=method_space)   # once, before the round loop
    ...
    # inside the round loop, right after you compute the global aggregate
    # and BEFORE/alongside your existing evaluation:
    smoothed = hook.update(global_state)               # global_state: {layer: (B,A)} or {layer: W}
    raw_acc  = evaluate(global_state, ...)             # your existing eval call, unchanged
    ema_acc  = evaluate(smoothed, ...)                 # same eval fn on the smoothed state
    round_log["acc_raw_eval"] = raw_acc
    round_log["acc_ema_eval"] = ema_acc

space per method:
    "ba"     — hetero_pad, hetlora : state is {layer: (B, A)} zero-padded to r_max
    "homo"   — homo_r8             : state is {layer: (B, A)} at fixed rank (same math)
    "deltaw" — flexlora            : state is {layer: W_agg}; EMA the W, then your
               existing eval path projects to eval_rank via truncated SVD as usual.

Notes:
  * beta=0.5 with bias correction — must match HetLoRA-M exactly or the control is invalid.
  * The HetLoRA row needs NO rerun: HetLoRA-M is HetLoRA + EMA-eval on an identical
    trajectory with shared seeds, so raw=41.5 / +EMA=43.2 is already a paired result.
  * AUC for the table = mean of acc_ema_eval over rounds 1..20, averaged over seeds;
    paired t-test against acc_raw_eval per seed (scipy.stats.ttest_rel).
  * Sanity check: rerun raw AUCs should match your main-table numbers within ~1 std.
    Report both if they drift; do not silently substitute.
"""

import torch


class EMAEvalHook:
    def __init__(self, beta: float = 0.5, space: str = "ba"):
        assert space in ("ba", "homo", "deltaw")
        self.beta = beta
        self.space = space
        self.t = 0
        self._ema = None

    @torch.no_grad()
    def update(self, global_state: dict) -> dict:
        """Feed the round-t raw global state; returns the bias-corrected EMA state.

        Accepts {layer: tensor} (deltaw) or {layer: (B, A)} (ba/homo).
        Does not modify global_state; the caller must keep distributing the RAW
        state to clients — the EMA output is for evaluation only.
        """
        self.t += 1
        b = self.beta
        if self._ema is None:
            self._ema = self._map(global_state, lambda x: (1 - b) * x.detach().clone())
        else:
            self._ema = self._zip(self._ema, global_state,
                                  lambda e, r: b * e + (1 - b) * r.detach())
        c = 1.0 - b ** self.t
        return self._map(self._ema, lambda x: x / c)

    # -- helpers handling both tensor and (B, A) tuple values -----------------
    @staticmethod
    def _map(state, fn):
        return {k: fn(v) if torch.is_tensor(v) else tuple(fn(x) for x in v)
                for k, v in state.items()}

    @staticmethod
    def _zip(s1, s2, fn):
        out = {}
        for k in s1:
            v1, v2 = s1[k], s2[k]
            if torch.is_tensor(v1):
                out[k] = fn(v1, v2)
            else:
                out[k] = tuple(fn(a, b) for a, b in zip(v1, v2))
        return out


# ---------------------------------------------------------------------------
# Run plan for the control (Yelp alpha=0.1 only):
#
#   methods needing rerun : flexlora (critical), hetero_pad, homo_r8
#   methods NOT rerun     : hetlora (row = existing HetLoRA vs HetLoRA-M numbers),
#                           spa_m   (excluded; its momentum state IS its eval model)
#
#   5 seeds: 15 runs ≈ 50 GPU-h ≈ ~1.5 days on 2x A6000
#   3 seeds:  9 runs ≈ 30 GPU-h ≈ overnight (note seed count in table caption)
#
#   for seed in 42 43 44 45 46; do for m in flexlora hetero_pad homo_r8; do
#     nohup python experiments/run_yelp.py --method $m --alpha 0.1 --seed $seed \
#       --ema_eval --device cuda:$((seed % 2)) >> logs/emactl_${m}_s${seed}.log 2>&1
#   done; done
#
# Going forward, also add a --save_states flag to fl_server.py (adapters are small);
# every future control/analysis becomes offline and free.
# ---------------------------------------------------------------------------
