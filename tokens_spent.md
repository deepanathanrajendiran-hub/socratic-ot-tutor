# Token spend — 6-turn live scenario

- **Generated:** 2026-05-02T09:18:07.723431+00:00
- **Backend:** http://localhost:8000
- **Session id:** `token-budget-1777713390`
- **Active domain:** `physics`
- **Primary model (Sonnet):** `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- **Fast model (Haiku):** `us.anthropic.claude-haiku-4-5-20251001-v1:0`

## Pricing (USD per 1M tokens, list)

| Model | Input | Output | Cache read |
|---|---|---|---|
| Sonnet | $3.00 | $15.00 | $0.30 |
| Haiku  | $0.80 | $4.00 | $0.08 |

## Per-turn detail

### Turn 0

**Student:** What's the law that says an object at rest stays at rest?

**Tutor:**

> Before we identify which principle describes that, let's start with what you've observed. Think about everyday situations — a book on a table, a hockey puck on ice, a car coasting down a straight road. What do you think determines whether something that's sitting still will start moving, or whether something already moving will speed up, slow down, or change direction?

**Pipeline steps:** concept_extraction → retrieval → classifier → generation → thinking → dean

**LLM calls this turn:**

| # | Model id | Family | Input | Output | Cache read | Cost |
|---|---|---|---:|---:|---:|---:|
| 0 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 916 | 43 | 0 | $0.00090 |
| 1 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 523 | 84 | 0 | $0.00075 |
| 2 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,512 | 4 | 0 | $0.00123 |
| 3 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 2,212 | 201 | 0 | $0.00965 |
| 4 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 4,543 | 34 | 0 | $0.01414 |
| **Σ** | | | **9,706** | **366** | **0** | **$0.0267** |

---

### Turn 1

**Student:** I think it has something to do with motion.

**Tutor:**

> You're right that we're talking about motion. But the key insight isn't just *that* objects move — it's about what happens when no forces act on them, or when forces are balanced. 
> 
> What do you think happens to an object's motion when nothing is pushing or pulling on it?

**Pipeline steps:** concept_extraction → retrieval → classifier → dean

**LLM calls this turn:**

| # | Model id | Family | Input | Output | Cache read | Cost |
|---|---|---|---:|---:|---:|---:|
| 0 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 530 | 91 | 0 | $0.00079 |
| 1 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,005 | 150 | 0 | $0.00140 |
| 2 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,599 | 4 | 0 | $0.00130 |
| 3 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 1,197 | 239 | 0 | $0.00718 |
| 4 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 2,615 | 34 | 0 | $0.00835 |
| **Σ** | | | **6,946** | **518** | **0** | **$0.0190** |

---

### Turn 2

**Student:** Maybe momentum?

**Tutor:**

> The answer is **Newton's first law**, also called the law of inertia. It states that an object at rest will remain at rest, and an object in motion will continue moving at a constant velocity in a straight line, unless acted upon by an external force. This principle explains why a book stays on a table until you push it, or why a rolling ball continues rolling until friction or another force stops it—the object's state of motion doesn't change on its own.
> 
> What would you like to do next?
> A) Try a clinical application question for this concept
> B) Move on to the next topic
> C) Stop here for now

**Pipeline steps:** concept_extraction → retrieval → classifier → dean

**LLM calls this turn:**

| # | Model id | Family | Input | Output | Cache read | Cost |
|---|---|---|---:|---:|---:|---:|
| 0 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 523 | 91 | 0 | $0.00078 |
| 1 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,054 | 150 | 0 | $0.00144 |
| 2 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,672 | 4 | 0 | $0.00135 |
| 3 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 1,229 | 250 | 0 | $0.00744 |
| 4 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 2,687 | 34 | 0 | $0.00857 |
| **Σ** | | | **7,165** | **529** | **0** | **$0.0196** |

---

### Turn 3

**Student:** Inertia maybe?

**Tutor:**

> Nice work on that one. Newton's first law is flagged for review if you'd like to strengthen that concept, or would you prefer to explore a different topic?

**Pipeline steps:** dean

**LLM calls this turn:**

| # | Model id | Family | Input | Output | Cache read | Cost |
|---|---|---|---:|---:|---:|---:|
| 0 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 155 | 4 | 0 | $0.00014 |
| 1 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 659 | 113 | 0 | $0.00367 |
| **Σ** | | | **814** | **117** | **0** | **$0.0038** |

---

### Turn 4

**Student:** It's Newton's first law.

**Tutor:**

> Exactly — [the target structure]. The heart of it is equilibrium: when forces on an object are balanced (the net force is zero), the object's velocity stays constant — which includes staying at rest if it started at rest. If you're pushing a stalled car on flat ground with constant velocity, what does that tell you about the forces acting on it?

**Pipeline steps:** concept_extraction → retrieval → classifier → generation → thinking → dean

**LLM calls this turn:**

| # | Model id | Family | Input | Output | Cache read | Cost |
|---|---|---|---:|---:|---:|---:|
| 0 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 341 | 4 | 0 | $0.00029 |
| 1 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 546 | 79 | 0 | $0.00075 |
| 2 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,089 | 150 | 0 | $0.00147 |
| 3 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,692 | 4 | 0 | $0.00137 |
| 4 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 2,252 | 255 | 1,532 | $0.01104 |
| 5 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 2,302 | 182 | 1,532 | $0.01010 |
| 6 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 4,219 | 300 | 0 | $0.01716 |
| **Σ** | | | **12,441** | **974** | **3,064** | **$0.0422** |

---

### Turn 5

**Student:** B

**Tutor:**

> You picked a menu option rather than answering the question. Let me rephrase: when you're pushing that stalled car at a steady speed across flat ground—not speeding up, not slowing down—the textbook tells us the system is in equilibrium. 
> 
> What does "equilibrium" mean about all the forces acting on the car?

**Pipeline steps:** concept_extraction → retrieval → classifier → dean

**LLM calls this turn:**

| # | Model id | Family | Input | Output | Cache read | Cost |
|---|---|---|---:|---:|---:|---:|
| 0 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,019 | 150 | 0 | $0.00142 |
| 1 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | haiku | 1,631 | 4 | 0 | $0.00132 |
| 2 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 3,133 | 426 | 0 | $0.01579 |
| 3 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 3,257 | 393 | 0 | $0.01567 |
| 4 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | sonnet | 4,218 | 34 | 0 | $0.01316 |
| **Σ** | | | **13,258** | **1,007** | **0** | **$0.0474** |

---

## Grand total

| Metric | Value |
|---|---|
| Input tokens | 50,330 |
| Output tokens | 3,511 |
| Cache-read tokens | 3,064 |
| **All tokens** | **56,905** |
| Total LLM calls | 29 |
| **Total cost** | **$0.1586** |
| Average per turn | $0.0264 |

## Demo budgeting

- **10 students × 1 scenario each**: ≈ $1.59
- **50 students × 1 scenario each**: ≈ $7.93
- **100 students × 1 scenario each**: ≈ $15.86
- **500 students × 1 scenario each**: ≈ $79.31

Add ~30% headroom for retries, exploratory chats, and Dean revisions that miss the prompt cache.
