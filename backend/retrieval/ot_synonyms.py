"""
retrieval/ot_synonyms.py — OT vocabulary expansion.

Maps lay terms, OT abbreviations, and clinical shorthand to anatomical
search terms. Zero latency — pure Python dict lookup, no LLM call.

Handles ~80% of vocabulary mismatch between student language
and textbook terminology without any API cost.

Usage:
    from retrieval.ot_synonyms import expand_query
    q = expand_query("Why is my funny bone so sensitive?")
    # → "Why is my funny bone so sensitive? ulnar nerve medial epicondyle"
"""

OT_SYNONYMS: dict[str, str] = {
    # ── Lay terms → anatomical terms ─────────────────────────────────────────
    "funny bone":             "ulnar nerve medial epicondyle",
    "wrist drop":             "radial nerve extensor paralysis posterior interosseous",
    "ape hand":               "median nerve thenar atrophy opposition loss",
    "claw hand":              "ulnar nerve intrinsic minus deformity",
    "drop foot":              "peroneal nerve dorsiflexion loss",
    "frozen shoulder":        "adhesive capsulitis glenohumeral joint",
    "rotator cuff":           "SITS supraspinatus infraspinatus teres subscapularis",
    "funny feeling pinky":    "ulnar nerve sensory distribution ring little finger",
    "weak grip":              "ulnar nerve hand intrinsic muscles hypothenar",
    "saturday night palsy":   "radial nerve compression spiral groove humerus",
    "carpal tunnel":          "median nerve transverse carpal ligament compression",
    "can't button shirt":     "fine motor median ulnar nerve coordination ADL",
    "can't open jar":         "grip strength ulnar nerve hypothenar power",

    # ── OT abbreviations ─────────────────────────────────────────────────────
    "ADL":   "activities of daily living functional independence",
    "ROM":   "range of motion joint mobility",
    "AROM":  "active range of motion voluntary",
    "PROM":  "passive range of motion",
    "FIM":   "functional independence measure ADL scoring",
    "SCI":   "spinal cord injury level function",
    "TBI":   "traumatic brain injury cognitive motor",
    "CVA":   "cerebrovascular accident stroke hemiplegia",
    "MCP":   "metacarpophalangeal joint finger knuckle",
    "PIP":   "proximal interphalangeal joint",
    "DIP":   "distal interphalangeal joint",
    "MMT":   "manual muscle testing strength grade",
    "NBCOT": "occupational therapy certification anatomy neuroscience",
    "OTPF":  "occupational therapy practice framework",

    # ── Spinal levels ─────────────────────────────────────────────────────────
    "C5 level": "C5 nerve root deltoid biceps shoulder abduction",
    "C6 level": "C6 nerve root wrist extension brachioradialis thumb",
    "C7 level": "C7 nerve root triceps wrist flexion middle finger",
    "C8 level": "C8 nerve root finger flexion intrinsic hand FDP",
    "T1 level": "T1 nerve root intrinsic hand muscles interossei lumbricals",

    # ── Clinical conditions ────────────────────────────────────────────────────
    "dupuytren":        "palmar fascia finger flexor contracture",
    "swan neck":        "PIP joint intrinsic muscles hyperextension",
    "cubital tunnel":   "ulnar nerve medial epicondyle elbow",
    "de quervain":      "abductor pollicis longus extensor pollicis brevis",
    "trigger finger":   "flexor tendon A1 pulley stenosing tenosynovitis",
    "jersey finger":    "flexor digitorum profundus distal phalanx avulsion",
    "tennis elbow":     "lateral epicondyle extensor carpi radialis overuse",
    "thoracic outlet":  "brachial plexus scalene muscles subclavian",
    "mallet finger":    "extensor tendon terminal slip distal phalanx",
    "gamekeeper thumb": "ulnar collateral ligament MCP joint thumb",
}


def expand_query(query: str) -> str:
    """
    Expand a student query with anatomical synonyms.

    Scans query for lay terms (case-insensitive substring match).
    Appends all matched anatomical term strings to the original query.
    The original query terms are always preserved.

    Args:
        query: original student question or concept string

    Returns:
        augmented query string (original + anatomical expansions)
    """
    query_lower = query.lower()
    additions: list[str] = []

    for lay_term, anatomical_terms in OT_SYNONYMS.items():
        if lay_term.lower() in query_lower:
            additions.append(anatomical_terms)

    if additions:
        return f"{query} {' '.join(additions)}"
    return query
