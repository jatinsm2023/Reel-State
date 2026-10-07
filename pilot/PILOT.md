# Reel State pilot: protocol and participant sheet

This file has two parts: the sheet you give participants, and the protocol you run. **Before recruiting anyone, show this to your course instructor and get their OK.** This is a student project collecting mood data from classmates; your department may have an ethics process that applies.

---

## Part 1: Participant information and consent (give this to each person)

**What this is.** A course project at IIT Kharagpur testing a movie recommender that takes your mood into account. It is not a medical or psychological tool and cannot diagnose or treat anything.

**What you'll do.** For about 10 days, open the app whenever you plan to watch something (at least 5 times in total). Each visit takes about 3 minutes: a few quick mood questions, an optional free-writing box, then film suggestions. After you watch, you tap how you feel now.

**What is recorded.**
* Your answers to the mood questions and how long you took to answer.
* *How* you type in the free-writing box: speed, pauses, corrections. **Never what you type.** The words stay in your browser and are discarded.
* The films suggested, which you said you watched, and how you felt afterwards.
* Your first name (or a nickname; you can use any name).

**What is not recorded.** The text you write, your location, your contacts, or anything else on your device.

**Your data.** It is kept on the researcher's computer, used only for this project, and reported only in anonymised form (P01, P02, …). **You can leave at any time and ask for everything about you to be deleted; it will be, the same day.** Just tell the researcher.

**Risks.** The questions are about everyday mood (energy, how your day went). If answering ever makes you uncomfortable, stop. If you are struggling with your mood, please talk to someone you trust or contact the institute's counselling service; this app is not a substitute.

**Consent.** By using the app after reading this, you confirm you are 18 or over, have read the above, and agree to take part.

---

## Part 2: Protocol (for you)

### Question and hypotheses (written down before data collection)

* **H1.** Within a person, the strategy that works better (match vs regulate) differs between people. *Evidence:* the spread of within-person (regulate − match) gains is wide, with some participants favouring each. (`pilot summary` prints this.)
* **H2.** The adaptive quiz asks fewer questions than a fixed 5-item form for a returning user. *Evidence:* mean questions per session after day 3, from `quiz_responses`.
* **H3.** Participants like the films it suggests. *Evidence:* the "Loved it" rate among check-ins, compared with 50%.

State up front what you will conclude if the result is null. With 10-15 people you will not get tight confidence intervals: report effect sizes and individual trajectories, not just p-values, and say plainly that the pilot is exploratory.

### Setup

1. Run the app: `scripts/serve.sh` (localhost only by default).
2. To let others reach it, put a tunnel in front of it (e.g. `cloudflared tunnel --url http://localhost:8000`). The tunnel gives you HTTPS. Share the link only with participants, and stop the tunnel when the pilot ends. Tokens protect each profile, but treat the link as private.
3. Tell each participant to open the link in the **same browser** every time (their profile and secret token live in that browser; clearing it loses the profile).

### Design notes

* The policy explores on its own (Thompson sampling), so each person naturally sees both strategies early on. Do not tell participants which strategy a suggestion used before they report how they feel (the screen shows it; ask them not to let it colour their answer, or hide the badge for the pilot).
* Ask for at least 5 check-ins each. Fewer than 3 make a participant uninformative for H1.
* Do not offer rewards that depend on liking the films.

### During and after

* `PYTHONPATH=src .venv/bin/python -m reelstate.pilot summary`: strategy outcomes and the within-person contrast.
* `PYTHONPATH=src .venv/bin/python -m reelstate.pilot export pilot.csv`: anonymised data for the report appendix.
* `PYTHONPATH=src .venv/bin/python -m reelstate.pilot delete <user_id>`: honour a deletion request.
* When finished, delete any remaining test profiles (yours included) before computing results, so they don't count as participants.

### What to report honestly

* How many people finished, and how many check-ins each.
* That "before" is the system's belief, not ground truth, and "after" is self-report.
* That participants are classmates (young, similar, probably tolerant of apps), so results do not generalise.
* That the simulator results show the mechanism works under stated assumptions; the pilot is the only evidence about real people.
