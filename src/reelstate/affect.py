"""Affect model: Russell's circumplex (valence x arousal) for movies, derived from the Tag Genome.

Each mood-relevant genome tag carries a hand-assigned (valence, arousal) pair in [-1, 1].
A movie's raw affect is the sum over its tags of relevance^3 * tag_affect (cubing makes strong
tags dominate and weak background relevance wash out); the catalog is then standardised and
squashed with tanh so the two axes are spread across the whole square instead of clumped.

The same table drives explanations ("recommended because: heartwarming, feel-good").
"""
import numpy as np

# tag -> (valence, arousal). Scores are judgement calls grounded in the circumplex model;
# they are documented here so the report can defend and ablate them.
TAG_AFFECT: dict[str, tuple[float, float]] = {
    # warm / pleasant, low-to-mid arousal
    "feel-good": (0.9, -0.1), "feel good movie": (0.9, -0.1), "heartwarming": (0.9, -0.2),
    "happy ending": (0.8, -0.1), "inspiring": (0.8, 0.2), "inspirational": (0.8, 0.2),
    "sweet": (0.8, -0.3), "cute": (0.8, -0.2), "cute!": (0.8, -0.2), "affectionate": (0.8, -0.3),
    "whimsical": (0.7, -0.1), "romantic": (0.7, 0.0), "romance": (0.5, 0.0), "love story": (0.6, 0.0),
    "love": (0.5, 0.0), "friendship": (0.7, 0.0), "family": (0.5, -0.1), "kids and family": (0.6, 0.0),
    "children": (0.5, 0.1), "christmas": (0.6, 0.0), "disney": (0.7, 0.1), "pixar": (0.7, 0.2),
    "beautiful": (0.7, -0.2), "beautiful scenery": (0.7, -0.3), "scenic": (0.5, -0.4), "touching": (0.4, -0.2),
    "nostalgic": (0.5, -0.4), "nostalgia": (0.5, -0.4), "light": (0.5, -0.3), "quirky": (0.5, 0.1),
    "good romantic comedies": (0.7, 0.1), "romantic comedy": (0.7, 0.1), "chick flick": (0.5, -0.1),
    "underdog": (0.6, 0.4), "rags to riches": (0.6, 0.3), "redemption": (0.5, 0.0), "courage": (0.5, 0.4),
    # funny, mid-to-high arousal
    "funny": (0.8, 0.3), "hilarious": (0.85, 0.4), "very funny": (0.85, 0.4), "funny as hell": (0.85, 0.5),
    "humorous": (0.7, 0.1), "comedy": (0.6, 0.2), "silly fun": (0.7, 0.2), "fun": (0.8, 0.4),
    "fun movie": (0.8, 0.4), "entertaining": (0.7, 0.3), "witty": (0.6, 0.2), "slapstick": (0.6, 0.6),
    "screwball": (0.6, 0.5), "parody": (0.4, 0.3), "spoof": (0.4, 0.3), "goofy": (0.6, 0.3),
    "silly": (0.5, 0.2), "farce": (0.4, 0.5), "over the top": (0.2, 0.6), "stoner movie": (0.4, 0.0),
    "dumb but funny": (0.4, 0.3), "sentimental": (0.3, -0.4), "sappy": (0.3, -0.3), "cheesy": (0.2, 0.0),
    # excitement
    "exciting": (0.6, 0.9), "action": (0.2, 0.8), "action packed": (0.3, 0.9), "fast paced": (0.2, 0.8),
    "adventure": (0.5, 0.7), "epic": (0.4, 0.6), "explosions": (0.1, 0.9), "car chase": (0.2, 0.9),
    "chase": (0.1, 0.8), "fight scenes": (0.0, 0.8), "gunfight": (-0.1, 0.8), "dynamic cgi action": (0.3, 0.8),
    "martial arts": (0.2, 0.7), "heist": (0.3, 0.7), "caper": (0.4, 0.5), "superhero": (0.4, 0.7),
    "super-hero": (0.4, 0.7), "space opera": (0.3, 0.6), "swashbuckler": (0.5, 0.7), "sword fight": (0.1, 0.8),
    "awesome": (0.7, 0.5), "twist ending": (0.1, 0.5), "plot twist": (0.1, 0.5), "mindfuck": (0.0, 0.6),
    # tension / dread
    "tense": (-0.3, 0.9), "intense": (-0.1, 0.9), "suspense": (-0.1, 0.8), "suspenseful": (-0.1, 0.8),
    "thriller": (-0.2, 0.7), "paranoia": (-0.5, 0.6), "claustrophobic": (-0.5, 0.5), "ominous": (-0.5, 0.4),
    "visceral": (-0.2, 0.8), "scary": (-0.5, 0.85), "frightening": (-0.6, 0.8), "horror": (-0.6, 0.8),
    "creepy": (-0.5, 0.5), "eerie": (-0.3, 0.2), "slasher": (-0.6, 0.8), "serial killer": (-0.6, 0.6),
    "murder": (-0.5, 0.5), "revenge": (-0.3, 0.6), "apocalypse": (-0.6, 0.6), "survival": (-0.2, 0.7),
    "war": (-0.4, 0.6), "war movie": (-0.4, 0.6), "violent": (-0.6, 0.8), "violence": (-0.6, 0.8),
    "bloody": (-0.5, 0.7), "gore": (-0.5, 0.7), "gory": (-0.5, 0.7), "gruesome": (-0.7, 0.6),
    "brutal": (-0.8, 0.6), "brutality": (-0.8, 0.6), "torture": (-0.9, 0.6), "gritty": (-0.5, 0.4),
    "disturbing": (-0.8, 0.4), "dark": (-0.6, 0.2), "dark comedy": (-0.1, 0.3), "black comedy": (-0.1, 0.2),
    # heavy / sad
    "depressing": (-0.9, -0.4), "bleak": (-0.9, -0.3), "downbeat": (-0.7, -0.4), "grim": (-0.8, 0.0),
    "heartbreaking": (-0.8, -0.1), "tragedy": (-0.8, 0.0), "sad": (-0.8, -0.4), "tear jerker": (-0.5, -0.1),
    "sad but good": (-0.4, -0.3), "melancholy": (-0.6, -0.6), "melancholic": (-0.6, -0.6), "bittersweet": (-0.2, -0.4),
    "hard to watch": (-0.8, 0.2), "depression": (-0.8, -0.5), "suicide": (-0.9, 0.1), "death": (-0.6, -0.1),
    "cancer": (-0.7, -0.2), "terminal illness": (-0.7, -0.3), "holocaust": (-0.9, 0.2), "genocide": (-0.9, 0.3),
    "child abuse": (-0.9, 0.3), "rape": (-0.9, 0.4), "loneliness": (-0.5, -0.5), "isolation": (-0.4, -0.3),
    "poverty": (-0.5, -0.2), "addiction": (-0.6, 0.2), "drug addiction": (-0.6, 0.2), "alcoholism": (-0.6, -0.1),
    "mental illness": (-0.5, 0.0), "insanity": (-0.4, 0.4), "dystopia": (-0.6, 0.3), "post apocalyptic": (-0.6, 0.4),
    "boring": (-0.5, -0.7), "unlikeable characters": (-0.4, 0.0), "moody": (-0.3, -0.3),
    # calm / contemplative
    "slow": (0.0, -0.8), "slow paced": (0.0, -0.8), "meditative": (0.2, -0.9), "reflective": (0.2, -0.7),
    "atmospheric": (0.1, -0.3), "dreamlike": (0.2, -0.5), "lyrical": (0.3, -0.6), "poetry": (0.2, -0.5),
    "understated": (0.1, -0.6), "character study": (0.0, -0.5), "dialogue driven": (0.1, -0.4), "talky": (0.0, -0.4),
    "wistful": (0.0, -0.6), "simple": (0.3, -0.4), "nature": (0.4, -0.4), "cerebral": (0.0, -0.3),
    "intimate": (0.3, -0.4), "thought-provoking": (0.1, -0.1), "philosophical": (0.0, -0.4), "deadpan": (0.2, -0.2),
}


def affect_scores(tags: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-movie (valence, arousal) in [-1, 1] from a movies x tags relevance matrix."""
    idx = {t: j for j, t in enumerate(tags)}
    cols, vals = [], []
    for tag, (v, a) in TAG_AFFECT.items():
        if tag in idx:
            cols.append(idx[tag])
            vals.append((v, a))
    sub = matrix[:, cols] ** 3                      # strong tags dominate
    w = np.asarray(vals, dtype=np.float32)          # (k, 2)
    raw = sub @ w                                   # (movies, 2)
    z = (raw - np.median(raw, axis=0)) / (raw.std(axis=0) + 1e-9)
    out = np.tanh(z / 1.5)
    return out[:, 0], out[:, 1]


def top_affect_tags(tag_rel: dict[str, float], k: int = 3) -> list[str]:
    """Tags that most explain a movie's affect, for 'why this movie' text."""
    scored = [(rel * (abs(TAG_AFFECT[t][0]) + abs(TAG_AFFECT[t][1])), t) for t, rel in tag_rel.items() if t in TAG_AFFECT]
    return [t for _, t in sorted(scored, reverse=True)[:k]]
