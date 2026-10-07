"""Item bank for the adaptive mood quiz.

Every item is a 5-option ordered question loading on ONE axis (valence or arousal). Options are
ordered from lowest to highest on that axis. IRT parameters (graded response model):
  a  discrimination: how sharply the item separates people near its thresholds
  b  four ordered thresholds on the latent z-scale (theta)

These are expert-set starting values. The report should state that they are priors, to be
re-calibrated from pilot responses (the quiz_responses table collects exactly that data).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Item:
    item_id: int
    axis: str                 # 'valence' | 'arousal'
    prompt: str
    options: tuple[str, ...]
    a: float
    b: tuple[float, float, float, float]


def _v(i, prompt, options, a, b):
    return Item(i, "valence", prompt, tuple(options), a, tuple(b))


def _a(i, prompt, options, a, b):
    return Item(i, "arousal", prompt, tuple(options), a, tuple(b))


BANK: tuple[Item, ...] = (
    # ------------------------------------------------------------------ valence
    _v(1, "Right now, how would you describe your overall mood?",
       ["Awful", "Low", "Okay", "Good", "Great"], 2.2, (-1.5, -0.6, 0.3, 1.2)),
    _v(2, "How did today treat you?",
       ["Terribly", "Rough", "Fine", "Pretty well", "Wonderfully"], 1.8, (-1.4, -0.5, 0.4, 1.3)),
    _v(3, "Over the last few hours, how much have you smiled or laughed?",
       ["Not at all", "Once or twice", "A bit", "Quite a lot", "Constantly"], 1.3, (-1.2, -0.3, 0.6, 1.5)),
    _v(4, "How do you feel about the rest of your evening?",
       ["Dreading it", "Not keen", "Neutral", "Looking forward to it", "Can't wait"], 1.5, (-1.3, -0.5, 0.5, 1.3)),
    _v(5, "If today were weather, what would it be?",
       ["Cold, heavy rain", "Grey and damp", "Mixed", "Mostly sunny", "Bright, clear sun"], 1.4, (-1.4, -0.6, 0.3, 1.2)),
    _v(6, "How much is on your mind that's bothering you?",
       ["A huge amount", "Quite a bit", "Some", "A little", "Nothing at all"], 1.6, (-1.4, -0.6, 0.3, 1.2)),
    _v(7, "How connected do you feel to the people in your life today?",
       ["Isolated", "Distant", "Neither", "Close", "Very close"], 1.0, (-1.3, -0.4, 0.5, 1.4)),
    _v(8, "Did anything go well for you today?",
       ["Nothing", "Hardly anything", "One small thing", "A few things", "Lots of things"], 1.2, (-1.2, -0.4, 0.4, 1.3)),
    _v(9, "How kind are you being to yourself right now?",
       ["Very harsh", "A bit harsh", "Neutral", "Fairly kind", "Very kind"], 1.1, (-1.2, -0.4, 0.5, 1.3)),
    _v(10, "How much are you enjoying the moment you're in?",
       ["Not at all", "Barely", "Somewhat", "Quite a lot", "Completely"], 1.7, (-1.3, -0.4, 0.5, 1.3)),
    # ------------------------------------------------------------------ arousal
    _a(11, "How much energy do you have right now?",
       ["Drained", "Low", "Moderate", "Energised", "Buzzing"], 2.1, (-1.4, -0.5, 0.4, 1.3)),
    _a(12, "How fast is your mind running?",
       ["Foggy and slow", "Slow", "Steady", "Quick", "Racing"], 1.8, (-1.3, -0.5, 0.5, 1.4)),
    _a(13, "How restless do you feel?",
       ["Completely settled", "Settled", "Neither", "Fidgety", "Can't sit still"], 1.6, (-1.2, -0.4, 0.5, 1.4)),
    _a(14, "How alert are you?",
       ["Exhausted", "Tired", "Okay", "Alert", "Wired"], 1.5, (-1.5, -0.6, 0.4, 1.3)),
    _a(15, "What pace do you want tonight to have?",
       ["Very slow and gentle", "Slow", "Relaxed", "Lively", "Fast and intense"], 1.4, (-1.3, -0.5, 0.4, 1.3)),
    _a(16, "Your heartbeat right now feels...",
       ["Very slow", "Calm", "Normal", "Quick", "Pounding"], 1.2, (-1.2, -0.4, 0.6, 1.5)),
    _a(17, "How much stimulation could you take right now?",
       ["None at all", "A little", "Some", "Plenty", "Bring it on"], 1.3, (-1.3, -0.5, 0.4, 1.3)),
    _a(18, "How noisy is it inside your head?",
       ["Silent", "Quiet", "Normal", "Noisy", "Chaos"], 1.4, (-1.3, -0.5, 0.5, 1.4)),
    _a(19, "How much do you feel like moving around?",
       ["Not at all", "A little", "Somewhat", "Quite a lot", "Desperately"], 1.1, (-1.2, -0.4, 0.5, 1.3)),
    _a(20, "How easily could you fall asleep right now?",
       ["Instantly", "Easily", "With some effort", "Hard", "Impossible"], 1.5, (-1.4, -0.5, 0.4, 1.3)),
)

BY_ID = {it.item_id: it for it in BANK}


def thresholds_array(item: Item) -> np.ndarray:
    return np.asarray(item.b, dtype=np.float64)
