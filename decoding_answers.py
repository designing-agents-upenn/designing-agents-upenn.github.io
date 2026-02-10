"""
Decoding Strategies for Language Generation
In-class exercise: Implement greedy, top-k, and top-p decoding.
Tokens are simplified to words. Use the vocabulary to look up P(word x | phrase y).
"""

from typing import Dict, List, Tuple
import random


# Vocabulary: maps phrase (tuple of words) -> {next_word: probability}
# Use this to look up P(word x | phrase y)
VOCAB = {
    ("the",): {"cat": 0.3, "dog": 0.3, "bird": 0.2, "fish": 0.2},
    ("the", "cat"): {"sat": 0.5, "is": 0.3, "was": 0.2},
    ("the", "cat", "is"): {"sleeping": 0.4, "black": 0.3, "soft": 0.3},
    ("the", "cat", "sat"): {"on": 0.6, "down": 0.3, "quietly": 0.1},
    ("the", "cat", "sat", "on"): {"the": 0.5, "a": 0.5},
    ("the", "cat", "sat", "on", "the"): {"mat": 0.6, "floor": 0.4},
    ("the", "cat", "sat", "on", "the", "mat"): {".": 0.5, "!": 0.3, "quietly": 0.2},
    ("the", "cat", "is", "sleeping"): {".": 0.5, "quietly": 0.3, "now": 0.2},
    ("the", "dog"): {"ran": 0.5, "is": 0.3, "barked": 0.2},
    ("the", "dog", "ran"): {"fast": 0.4, "away": 0.3, "quickly": 0.2, "home": 0.1},
    ("the", "dog", "ran", "fast"): {"!": 0.5, ".": 0.3, "through": 0.2},
    ("the", "fish"): {"swam": 0.5, "is": 0.3, "lives": 0.2},
    ("the", "fish", "swam"): {"away": 0.5, "fast": 0.3, "slowly": 0.2},
    ("the", "fish", "swam", "away"): {".": 0.6, "!": 0.4},
    ("the", "bird"): {"flew": 0.5, "is": 0.3, "sang": 0.2},
    ("the", "bird", "flew"): {"away": 0.5, "high": 0.3, "home": 0.2},
    ("the", "bird", "flew", "away"): {".": 0.6, "!": 0.4},
    ("hello",): {"world": 0.5, "there": 0.3, "friend": 0.2},
    ("hello", "world"): {"!": 0.6, ".": 0.3, "says": 0.1},
}


def get_next_word_probs(phrase: Tuple[str, ...]) -> Dict[str, float]:
    """
    Look up the probability distribution for the next word given a phrase.
    Returns a dict mapping each possible next word to its probability.
    Returns empty dict if phrase is not in vocabulary (end of generation).
    """
    return dict(VOCAB.get(phrase, {}))


def greedy_decode(start_phrase: List[str], max_length: int = 10) -> List[str]:
    """
    Greedy decoding: at each step, pick the word with highest probability.
    """
    phrase = list(start_phrase)
    while len(phrase) < max_length:
        context = tuple(phrase)
        probs = get_next_word_probs(context)
        if not probs:
            break
        next_word = max(probs, key=probs.get)
        phrase.append(next_word)
    return phrase


def top_k_decode(
    start_phrase: List[str], k: int = 2, max_length: int = 10, seed: int = 42
) -> List[str]:
    """
    Top-k decoding: at each step, consider only the top k words by probability,
    renormalize their probabilities, and sample from that distribution.
    """
    random.seed(seed)
    phrase = list(start_phrase)
    while len(phrase) < max_length:
        context = tuple(phrase)
        probs = get_next_word_probs(context)
        if not probs:
            break
        # Sort by probability descending and take top k
        sorted_words = sorted(probs, key=probs.get, reverse=True)
        top_k_words = sorted_words[:k]
        top_k_probs = {w: probs[w] for w in top_k_words}
        total = sum(top_k_probs.values())
        # Renormalize
        top_k_probs = {w: p / total for w, p in top_k_probs.items()}
        words, weights = zip(*top_k_probs.items())
        next_word = random.choices(words, weights=weights, k=1)[0]
        phrase.append(next_word)
    return phrase


def top_p_decode(
    start_phrase: List[str], p: float = 0.8, max_length: int = 10, seed: int = 42
) -> List[str]:
    """
    Top-p (nucleus) decoding: at each step, take the smallest set of words
    whose cumulative probability >= p, renormalize, and sample from that set.
    """
    random.seed(seed)
    phrase = list(start_phrase)
    while len(phrase) < max_length:
        context = tuple(phrase)
        probs = get_next_word_probs(context)
        if not probs:
            break
        # Sort by probability descending
        sorted_words = sorted(probs, key=probs.get, reverse=True)
        cumulative = 0.0
        nucleus_words = []
        for w in sorted_words:
            cumulative += probs[w]
            nucleus_words.append(w)
            if cumulative >= p:
                break
        nucleus_probs = {w: probs[w] for w in nucleus_words}
        total = sum(nucleus_probs.values())
        nucleus_probs = {w: prob / total for w, prob in nucleus_probs.items()}
        words, weights = zip(*nucleus_probs.items())
        next_word = random.choices(words, weights=weights, k=1)[0]
        phrase.append(next_word)
    return phrase


def main():
    start_phrase = ["the"]
    max_length = 8

    print("Start phrase:", " ".join(start_phrase))
    print()

    greedy_result = greedy_decode(start_phrase, max_length=max_length)
    print("Greedy decoding:")
    print("  ", " ".join(greedy_result))
    print()

    top_k_result = top_k_decode(start_phrase, k=3, max_length=max_length)
    print("Top-k decoding (k=3):")
    print("  ", " ".join(top_k_result))
    print()

    top_p_result = top_p_decode(start_phrase, p=0.9, max_length=max_length)
    print("Top-p decoding (p=0.9):")
    print("  ", " ".join(top_p_result))


if __name__ == "__main__":
    main()
