"""
Install:
  pip install fastembed

FastEmbed docs (TextEmbedding, embed): https://github.com/qdrant/fastembed
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


# -----------------------------
# Restaurant recommendation corpus (provided)
# -----------------------------
chunks: Dict[str, str] = {
    # Los Angeles (10)
    "LA_01": "Cafe Gratitude is in Los Angeles, open until 11pm, and serves fully vegan and health-conscious comfort food.",
    "LA_02": "Night Market Noodles is in Los Angeles, open until 1am, and offers vegetarian and gluten-free rice noodle bowls.",
    "LA_03": "Silver Lake Diner is in Los Angeles, open 24 hours, and has classic diner options plus dairy-free milkshakes.",
    "LA_04": "Sunset Grill House is in Los Angeles, open until 10pm, and features halal grilled chicken plates and salads.",
    "LA_05": "Marina Poke Bar is in Los Angeles, open until 9pm, and has pescatarian bowls with low-carb and keto add-ons.",
    "LA_06": "Little Tokyo Bento Spot is in Los Angeles, open until 8pm, and includes vegetarian bento boxes and miso soup.",
    "LA_07": "Westwood Falafel Kitchen is in Los Angeles, open until midnight, and serves vegan falafel wraps and nut-free sides.",
    "LA_08": "Echo Park Taqueria is in Los Angeles, open until 2am, and has meat tacos, vegan cactus tacos, and corn tortillas.",
    "LA_09": "Venice Garden Cafe is in Los Angeles, open until 6pm, and focuses on organic vegetarian brunch and fresh juices.",
    "LA_10": "Downtown Ramen Lab is in Los Angeles, open until 11pm, and offers pork broth plus a vegan mushroom broth.",
    # New York City (10)
    "NYC_01": "Greenwich Village Kitchen is in New York City, open until 10pm, and serves vegetarian pasta with gluten-free noodles.",
    "NYC_02": "Midtown Bento Express is in New York City, open until 9pm, and has halal teriyaki chicken and tofu rice boxes.",
    "NYC_03": "Lower East Side Deli is in New York City, open until 7pm, and offers kosher sandwiches and dairy-free soups.",
    "NYC_04": "Harlem Spice Table is in New York City, open until midnight, and features vegan curry and spicy seafood stew.",
    "NYC_05": "Queens Dumpling Corner is in New York City, open until 11pm, and serves pork dumplings, vegetable dumplings, and peanut-free sauces.",
    "NYC_06": "Brooklyn Brick Oven is in New York City, open until 1am, and has wood-fired pizza with vegan cheese options.",
    "NYC_07": "SoHo Salad Studio is in New York City, open until 8pm, and specializes in keto bowls and high-protein add-ons.",
    "NYC_08": "Upper West Vegan Table is in New York City, open until 9pm, and serves fully plant-based comfort dishes.",
    "NYC_09": "Chinatown Noodle Hall is in New York City, open until 2am, and offers hand-pulled noodles with vegetarian broth choices.",
    "NYC_10": "Astoria Grill Room is in New York City, open until 10pm, and includes Mediterranean seafood, halal lamb, and gluten-free sides.",
    # Philadelphia (10)
    "PHL_01": "Old City Tavern Eats is in Philadelphia, open until 11pm, and serves gastropub plates with vegetarian burger options.",
    "PHL_02": "University City Wraps is in Philadelphia, open until 9pm, and offers vegan wraps, halal chicken, and nut-free dressings.",
    "PHL_03": "Fishtown Noodle Garage is in Philadelphia, open until midnight, and has ramen with pork, chicken, or vegan broth.",
    "PHL_04": "South Philly Pizza Co is in Philadelphia, open until 1am, and serves late-night slices with cauliflower gluten-free crust.",
    "PHL_05": "Rittenhouse Garden Bistro is in Philadelphia, open until 10pm, and focuses on vegetarian small plates and seasonal produce.",
    "PHL_06": "Northern Liberties BBQ is in Philadelphia, open until 10pm, and has smoked meats plus a vegan jackfruit platter.",
    "PHL_07": "Center City Salad Bar is in Philadelphia, open until 6pm, and specializes in keto salads and dairy-free dressings.",
    "PHL_08": "West Philly Curry House is in Philadelphia, open until 11pm, and serves halal curries with vegan lentil choices.",
    "PHL_09": "Reading Terminal Bowl Shop is in Philadelphia, open until 5pm, and offers gluten-free grain bowls and pescatarian toppings.",
    "PHL_10": "Manayunk Brunch Spot is in Philadelphia, open until 3pm, and features vegetarian omelets and lactose-free smoothie options.",
}


# -----------------------------
# Embedding helpers (provided)
# -----------------------------

def _embed_texts_fastembed(texts: List[str]) -> np.ndarray:
    """Returns embeddings as np.ndarray [n, d], L2-normalized."""
    model = TextEmbedding(model_name=DEFAULT_MODEL)
    vectors = list(model.embed(texts))
    X = np.array(vectors, dtype=np.float32)
    return _l2_normalize_rows(X)


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


# -----------------------------
# STUDENT TODOs (implement only these 2)
# -----------------------------

def build_embedding_store(chunks: Dict[str, str]) -> Dict[str, Any]:
    """
    TODO (1/2): Build an embedding store.

    Do this:
      - Use a stable order of IDs (e.g., sorted(chunks)).
      - Embed all chunk texts.
      - Return a dict with:
          "chunk_ids", "texts", "embeddings"
    """
    raise NotImplementedError


def retrieve(
    query: str, store: Dict[str, Any], top_k: int = 5
) -> List[Tuple[str, str, float]]:
    """
    TODO (2/2): Retrieve top_k chunks.

    Do this:
      - Embed the query with FastEmbed.
      - Score against store["embeddings"] with cosine similarity
        (dot product because vectors are L2-normalized).
      - Return top_k tuples: (chunk_id, chunk_text, score), sorted high to low.
    """
    raise NotImplementedError


# -----------------------------
# Demo runner (provided)
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    store = build_embedding_store(chunks)
    demo_queries = [
        "I need a vegan place in Los Angeles open late.",
        "Find a halal dinner option in New York City.",
        "What is a gluten-free place in Philadelphia?",
    ]

    for q in demo_queries:
        print(f"\nQuery: {q}")
        hits = retrieve(q, store, top_k=args.top_k)
        for chunk_id, _, score in hits:
            print(f"{score:.4f}\t{chunk_id}")


if __name__ == "__main__":
    main()