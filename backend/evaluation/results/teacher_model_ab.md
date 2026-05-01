# Teacher-model A/B — Sonnet 4.5 vs Haiku 4.5

_Generated_: `2026-04-30T06:53:07.072102+00:00`

- Concept: `ulnar nerve`
- Scenarios: **7**
- Models: **sonnet-4.5** (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`), **haiku-4.5** (`us.anthropic.claude-haiku-4-5-20251001-v1:0`)

Both models run on the SAME prompt template, SAME chunks, SAME scenarios. The faithfulness judge is a separate Haiku call — kept constant across both teacher models so the scoring is consistent.

## Summary

| Metric | sonnet-4.5 | haiku-4.5 | better |
|---|---|---|---|
| Premature reveal rate (lower better) | 0% | 0% | tie |
| Ends with `?` rate (higher better) | 100% | 100% | tie |
| Faithfulness (higher better) | 0.88 | 0.81 | sonnet |
| Avg response chars | 356.4 | 499.6 | — |
| Avg latency (ms, lower better) | 8623 | 3857 | haiku |

## Per-scenario rows

### S1: Open question (T0)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 374 chars · 8737 ms · 3/3 supported (100%)

> That "funny bone" sensation happens when you bump a specific spot at your elbow—right behind that bony prominence on the inner side. A nerve runs very superficially there, just under the skin, which is why hitting it creates that characteristic tingling down into your hand.
> 
> Can you tell me which fingers typically feel that tingling sensation when you hit your funny bone?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 672 chars · 4116 ms · 2/2 supported (100%)

> Great question—that's a perfect clinical observation to start with! When you hit your "funny bone," you're actually striking the bony prominence on the inside of your elbow, and the tingling sensation that shoots down your forearm and into your hand tells us something important: a major nerve must be running right there, very close to the surface.
> 
> Here's what I want you to think about: **that nerve runs in a groove directly behind the medial epicondyle of the humerus—almost like it's sitting in a tunnel at the elbow.** 
> 
> Can you think about which of the major forearm nerves would be positioned there, and what sensory territory it would supply if you irritated it?

---

### S2: I don't know (T0)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 365 chars · 8907 ms · 2/2 supported (100%)

> I understand it's tempting to jump straight to the answer, but working through this will help you retain it much better for clinical practice. Let me give you a starting point: this nerve arises from the C8 and T1 nerve roots and comes from the medial cord of the brachial plexus.
> 
> What major nerves do you know that originate from that part of the brachial plexus?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 543 chars · 3959 ms · 2/2 supported (100%)

> I hear you—anatomy can feel overwhelming! But here's the thing: if I just hand you the answer, you won't be able to recognize this nerve's injury pattern when you see it in a client, and that's where the real skill lives.
> 
> Let me make this easier. Imagine a client comes to your clinic with weakness in their hand grip and a characteristic "claw" appearance in their ring and pinky fingers. **That's a clinical presentation you need to recognize.**
> 
> What do you already know about where nerves can get pinched or compressed in the arm or hand?

---

### S3: Wrong guess — median nerve (T0)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 363 chars · 8342 ms · 2/3 supported (67%)

> Not quite! The median nerve actually supplies sensation to the thumb, index, middle, and the *lateral* half of the ring finger – essentially the thumb side of the hand. 
> 
> Think about which nerve would be responsible for sensation to the *little finger* and the medial (ulnar) side of the hand. What nerve travels posterior to the medial epicondyle of the humerus?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 305 chars · 2926 ms · 3/3 supported (100%)

> Not quite—let me give you a hint to reconsider. Think about which nerve passes *behind* the medial epicondyle of the humerus and, when compressed in the cubital tunnel, produces that characteristic **claw-hand deformity** with weakness of the small hand muscles. 
> 
> Which nerve do you think that describes?

---

### S4: Still no idea (T1)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 471 chars · 10259 ms · 2/2 supported (100%)

> I understand this is challenging! Let me give you a helpful landmark: this nerve passes right behind the medial epicondyle of the humerus — you know, that spot on the inside of your elbow where you get that "funny bone" tingling sensation when you bump it. That tingling actually happens because you're compressing this particular nerve against the bone.
> 
> Now, thinking about that sensation when you hit your funny bone — which fingers does that tingling shoot down into?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 489 chars · 3939 ms · 3/3 supported (100%)

> I hear you — let's take a step back and use something more concrete. When this nerve gets compressed at the elbow (in a space called the cubital tunnel), it causes a very distinctive hand deformity where the fingers curl inward in a claw-like position, especially the ring and pinky fingers. This happens because the small muscles between the bones of the hand lose their nerve supply.
> 
> **Does that claw-hand deformity ring a bell from anything you've seen in clinic or in your readings?**

---

### S5: Wrong region — wrist nerve (T1)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 316 chars · 8963 ms · 3/3 supported (100%)

> You're on the right track thinking about the wrist area, but this particular nerve has a more famous landmark higher up in the arm. Think about where you might bump your elbow and get that "funny bone" tingling sensation that shoots down into your hand.
> 
> Which bony landmark at the elbow does this nerve pass behind?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 473 chars · 4296 ms · 2/2 supported (100%)

> Good instinct—there are indeed several nerves that pass through the wrist region. But let me ask you this: **which specific fingers are numb?** That's actually your most important clue. Different nerves in the forearm and hand have very distinct sensory territories, and if you can map out which fingers are affected, you'll narrow down which nerve we're dealing with pretty quickly.
> 
> So tell me—is it the thumb side of the hand, the pinky side, or somewhere in the middle?

---

### S6: Anatomy clarifying question (T0)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 388 chars · 7248 ms · 2/2 supported (100%)

> This nerve provides sensation to the little finger and the ulnar half of the ring finger—so you can think of it as serving the "pinky side" of the hand. That's the sensory distribution, which is really helpful for identifying nerve injuries clinically.
> 
> Now, thinking about the motor function: what kind of hand deformity might you expect to see if this nerve were chronically compressed?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 631 chars · 3860 ms · 5/5 supported (100%)

> This nerve innervates the **little finger** and the **medial half of the ring finger**—so you're looking at the ulnar side of the hand. That sensory distribution is really important clinically because patients often report tingling or numbness in those specific fingers when there's a problem.
> 
> **Breadcrumb:** The nerve that does this originates from the lower cervical and first thoracic nerve roots (C8 and T1), which makes it part of the medial cord of the brachial plexus.
> 
> Now, when you think about motor control in the hand, what happens to grip strength and finger positioning if this nerve becomes compressed at the elbow?

---

### S7: Wrong cord level (T1)

**sonnet-4.5** — ✅ no-leak · ends-with-?: ✅ · 218 chars · 7909 ms · 1/2 supported (50%)

> Not quite — C5 and C6 are actually the upper roots of the brachial plexus. This nerve comes from much lower in the plexus, from the medial cord. 
> 
> What are the lowest nerve roots that contribute to the brachial plexus?

**haiku-4.5** — ✅ no-leak · ends-with-?: ✅ · 384 chars · 3905 ms · 0/4 supported (0%)

> Not quite — C5 and C6 actually form the **upper trunk** of the brachial plexus, and they give rise to nerves like the axillary and musculocutaneous. The nerve we're discussing here comes from the **lower trunk**, which is formed by the lower cervical and first thoracic roots.
> 
> Given that this nerve is associated with the lower trunk, which nerve roots do you think contribute to it?

---
