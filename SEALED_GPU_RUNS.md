# Sealed-noise accuracy runs (GPU)

These runs measure model accuracy when every client release goes through the
sealed-noise box: the whole update (all q_proj and v_proj modules) is clipped
to norm C and Gaussian noise is added, calibrated to (eps, 1e-5)-DP per release.
Method: FlexLoRA (the server receives full ΔW, which is exactly what the box releases).

New code: `src/privacy/sealed_release.py`, the `sealed_noise=` argument of
`run_federated` in `src/server/fl_server.py`, and `experiments/run_sealed_utility.py`.
The full training loop has not been run with this hook yet (no GPU on the laptop),
so start with the smoke test.

## 1. Smoke test (a few minutes)

```bash
cd rework
python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions clip 100 --num-rounds 1 --results-dir results_sealed_smoke
```

Check: it finishes, `results_sealed_smoke/yelp/100/sealed_100_seed42.json` exists,
and the log lines say `sealed release norm=... clipped=... eps_client=...`.

## 2. Main sweep (Yelp, one seed first)

```bash
python experiments/run_sealed_utility.py --dataset yelp --seed 42
```

Conditions run in order: `none` (ordinary training), `clip` (clipping only),
then eps per release = 1e5, 1e4, 1e3, 100, 10, 1. Each condition is a full
20-round run, so this is 8 runs, about as long as 8 runs of the earlier campaign.
Finished conditions are skipped if you restart.

Then, if time allows, seeds 43 and 44, and GSM8K:

```bash
python experiments/run_sealed_utility.py --dataset yelp --seed 43
python experiments/run_sealed_utility.py --dataset yelp --seed 44
python experiments/run_sealed_utility.py --dataset gsm8k --seed 42
```

## 3. One run with the full protocol in the loop (optional)

Routes every release through the FedGT box (attestation, nonce, signature,
encryption, server checks). Slower: about 1 to 2 minutes of CPU per release.

```bash
python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions 100 --impl box --results-dir results_sealed_box
```
Needs `pip install cryptography` and the FedGT folder next to rework/.

## What to send back

The folder `results_sealed/` (JSON only, small). I will make the accuracy-vs-privacy figure from it.

## What to expect

With dense updates of 411M entries and 5 clients per round, noise at meaningful
privacy (eps per release around 1 to 10) is thousands of times larger than the
update, so accuracy will likely stay at the base model's level. That is a real
result, not a bug: it is the price of local DP for a 7B model, and it is the
trade-off the reviewer asked us to measure. The sweep shows where learning starts.
