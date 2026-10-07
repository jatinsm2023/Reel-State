# Final simulation results (fresh seeds)

60 synthetic users x 40 days x seeds [11, 12, 13]. Gain = true change in wellbeing, noise-free, in [-1, 1]. Intervals are 95% t-intervals over seeds. Policy hyperparameters were tuned on a separate seed (99).

## Scenario: No heterogeneity (changing mood always wins)

| Policy | Gain, last 10 days | Gain, all days |
|---|---|---|
| Always match | +0.024 ± 0.039 | +0.021 ± 0.023 |
| Random | +0.146 ± 0.016 | +0.145 ± 0.018 |
| Learned (no pooling) | +0.221 ± 0.008 | +0.206 ± 0.018 |
| Learned (pooled) | +0.228 ± 0.032 | +0.218 ± 0.013 |
| Always regulate | +0.258 ± 0.018 | +0.255 ± 0.005 |

* Personalised-oracle ceiling (all days): **+0.255 ± 0.005**; best single fixed strategy: +0.255 ± 0.005.
* **Headroom** (most that personalisation could add): **+0.000 ± 0.000**.
* **Exploration cost** (always-regulate minus learned, all days): **+0.037 ± 0.013**.

* Pooling ablation (pooled minus flat, all days): +0.012 ± 0.007.

## Scenario: Strong heterogeneity (a minority needs matching)

| Policy | Gain, last 10 days | Gain, all days |
|---|---|---|
| Always match | +0.016 ± 0.026 | +0.018 ± 0.025 |
| Random | +0.100 ± 0.025 | +0.100 ± 0.006 |
| Learned (no pooling) | +0.148 ± 0.024 | +0.142 ± 0.037 |
| Learned (pooled) | +0.158 ± 0.025 | +0.146 ± 0.032 |
| Always regulate | +0.179 ± 0.020 | +0.180 ± 0.041 |

* Personalised-oracle ceiling (all days): **+0.215 ± 0.031**; best single fixed strategy: +0.180 ± 0.041.
* **Headroom** (most that personalisation could add): **+0.035 ± 0.014**.
* **Exploration cost** (always-regulate minus learned, all days): **+0.034 ± 0.012**.

* Pooling ablation (pooled minus flat, all days): +0.004 ± 0.009.

## Horizon study (strong heterogeneity, 30 users)

Per-session gain of learned minus always-regulate, by period:

| Days | learned − always_regulate |
|---|---|
| 1–30 | -0.029 ± 0.014 |
| 31–60 | -0.015 ± 0.023 |
| 61–90 | -0.002 ± 0.021 |
| 91–120 | -0.007 ± 0.023 |

* Cumulative break-even day: **not reached within 120 days**.

Share of low-mood evenings where the learned policy picks MATCH (last 30 days of the horizon run):

| Hidden type | Share picking match |
|---|---|
| cathartic | 0.45 ± 0.19 |
| regulator | 0.11 ± 0.12 |
| neutral | 0.14 ± 0.11 |
