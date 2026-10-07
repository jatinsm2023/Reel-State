# Shipped policy (switch margin 0.08) on fresh seeds

Same seeds and random draws as the baselines in `sim_summary_v2.md`; margin tuned on throwaway seed 99.

## No heterogeneity (changing mood always wins)

| Policy | Gain, last 10 days | Gain, all days | Gap to always-regulate (all days) |
|---|---|---|---|
| Always match | +0.024 ± 0.039 | +0.021 ± 0.023 | -0.234 ± 0.018 |
| Learned, margin 0 | +0.228 ± 0.032 | +0.218 ± 0.013 | -0.037 ± 0.013 |
| **Learned, margin 0.08** | +0.247 ± 0.018 | +0.237 ± 0.014 | -0.018 ± 0.016 |
| Always regulate | +0.258 ± 0.018 | +0.255 ± 0.005 | +0.000 ± 0.000 |

Personalised-oracle ceiling (all days): +0.255.

## Strong heterogeneity (a minority needs matching)

| Policy | Gain, last 10 days | Gain, all days | Gap to always-regulate (all days) |
|---|---|---|---|
| Always match | +0.016 ± 0.026 | +0.018 ± 0.025 | -0.162 ± 0.066 |
| Learned, margin 0 | +0.158 ± 0.025 | +0.146 ± 0.032 | -0.034 ± 0.012 |
| **Learned, margin 0.08** | +0.172 ± 0.012 | +0.163 ± 0.042 | -0.017 ± 0.005 |
| Always regulate | +0.179 ± 0.020 | +0.180 ± 0.041 | +0.000 ± 0.000 |

Personalised-oracle ceiling (all days): +0.215.

## Horizon (strong heterogeneity, 30 users)

| Days | learned − always_regulate |
|---|---|
| 1–30 | -0.019 ± 0.018 |
| 31–60 | -0.007 ± 0.017 |
| 61–90 | +0.002 ± 0.022 |
| 91–120 | +0.005 ± 0.014 |

* Cumulative break-even day: **not reached within 120 days**.

Share of low-mood evenings where the policy picks MATCH (last 30 days of horizon):

| Hidden type | Share picking match |
|---|---|
| cathartic | 0.25 ± 0.17 |
| regulator | 0.04 ± 0.03 |
| neutral | 0.05 ± 0.01 |
