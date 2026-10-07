# Simulation results

60 synthetic users x 40 days x 3 seeds. Gain = true change in wellbeing (valence up, arousal less extreme), noise-free, range [-1, 1]. Intervals are 95% t-intervals over seeds.

## A. Does learning match-vs-regulate help?

| Policy | Gain, last 10 days | Gain, all days | Regret vs hindsight-best fixed |
|---|---|---|---|
| Learned policy (Reel State) | +0.210 ± 0.015 | +0.201 ± 0.016 | 0.065 ± 0.008 |
| Always match | +0.024 ± 0.003 | +0.029 ± 0.009 | 0.250 ± 0.017 |
| Always regulate | +0.266 ± 0.027 | +0.273 ± 0.013 | 0.009 ± 0.007 |
| Random | +0.151 ± 0.022 | +0.154 ± 0.017 | 0.124 ± 0.007 |

## Adaptation by hidden person type (low-valence evenings, last 15 days)

| Hidden type | Share of picks that were 'match' |
|---|---|
| regulator | 0.24 ± 0.07 |
| cathartic | 0.26 ± 0.15 |
| neutral | 0.29 ± 0.09 |

## B. Which sensors earn their place?

| Sensing | Mood error (VA distance) | Questions / session | Gain, last 10 days |
|---|---|---|---|
| Context only | 0.615 ± 0.005 | 0.00 ± 0.00 | +0.129 ± 0.006 |
| Typing + context | 0.572 ± 0.010 | 0.00 ± 0.00 | +0.135 ± 0.060 |
| Fixed 5-question quiz | 0.364 ± 0.025 | 5.00 ± 0.00 | +0.223 ± 0.061 |
| Adaptive quiz | 0.387 ± 0.016 | 4.08 ± 0.06 | +0.212 ± 0.042 |
| Everything fused | 0.398 ± 0.001 | 2.86 ± 0.03 | +0.210 ± 0.015 |

## C. Cold start: does borrowing the population's experience help new users?

| Variant | Gain, first 7 days | Gain, last 10 days |
|---|---|---|
| With population prior | +0.184 ± 0.008 | +0.210 ± 0.015 |
| Without | +0.169 ± 0.039 | +0.197 ± 0.006 |

## D. Does the 'people feeling like you' signal help?

| Variant | Gain, all days | Gain, last 10 days |
|---|---|---|
| With similar-mood boost | +0.201 ± 0.016 | +0.210 ± 0.015 |
| Without | +0.189 ± 0.010 | +0.196 ± 0.027 |
