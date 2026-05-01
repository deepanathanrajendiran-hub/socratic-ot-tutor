# Socratic-OT — Physics generalizability E2E report

- **Generated:** 2026-04-30T05:50:49.965691+00:00
- **Backend:** http://localhost:8000
- **Health:** `{'status': 'ok', 'version': '0.1.0'}`

## Summary — 10 / 10 scenarios passed

- ✅ **PA1 — Cooperative trajectory (Newton's second law)**
- ✅ **PA2 — Cooperative trajectory (kinetic energy)**
- ✅ **PA3 — Cooperative trajectory (projectile motion)**
- ✅ **PB1 — Wrong attempt (force = mass)**
- ✅ **PC1 — Three IDKs → reveal of Newton's second law**
- ✅ **PD1 — Off-topic injection (restaurant)**
- ✅ **PG1 — Vague opener ('tell me about physics')**
- ✅ **PI1 — First-try mastery menu (kinetic energy)**
- ✅ **PL2 — Wrong mechanism (force = velocity)**
- ✅ **PW1 — IDK reveal seeds weak_topics**

## PA1 — Cooperative trajectory (Newton's second law)

_Open-ended physics question → progressive scaffolding → correct answer. Verifies turn-0 no-reveal + breadcrumb pacing on the new domain._

- session_id: `e2e-b6db5a075d`
- result: **PASS**

### Turn 0

**Student:** What does Newton's second law actually say?

**Tutor:**

> Before we explore that, let's start with what you already understand. When you push a shopping cart or throw a ball, what do you think determines how quickly it speeds up or slows down?

_(turn_count after: 1)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `no meta-leak` — no meta phrases
- ✅ `ends with question` — ends with a question
- ✅ `≤ 2 question marks` — 1 '?' in response
- ✅ `no concept reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`CORRECT`, duration=1949ms
  - retrieval-out: `{"crag_decision": "CORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we explore that, let's start with what you already understand. When you push a shopping cart or throw a ball, what do you think determines how quickly it speeds up or slows down?
    ```

</details>

### Turn 1

**Student:** Force equals mass times acceleration?

**Tutor:**

> You've written down a mathematical relationship. But before we settle on whether that captures the full picture, let's think about what each piece means. When you see that equation, you're relating three quantities to one another.
> 
> What happens to the acceleration if you double the mass but keep the force the same?

_(turn_count after: 2)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `non-empty response` — response length: 316 chars

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1956ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=1 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    You've written down a mathematical relationship. But before we settle on whether that captures the full picture, let's think about what each piece means. When you see that equation, you're relating three quantities to one another.
    
    What happens to the acceleration if you double the mass but keep the force the same?
    ```

</details>

---

## PA2 — Cooperative trajectory (kinetic energy)

_Different physics concept — verifies the lock generalizes._

- session_id: `e2e-5035b7a7a8`
- result: **PASS**

### Turn 0

**Student:** What's the formula that relates an object's mass and speed to its energy of motion?

**Tutor:**

> Before we jump to formulas, let's start with what you already know. When you see something moving — say, a car rolling down the street or a ball flying through the air — what do you think determines how much energy that motion carries?

_(turn_count after: 1)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `ends with question` — ends with a question
- ✅ `no concept reveal` — no discriminating part of 'kinetic energy' present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `kinetic energy`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`AMBIGUOUS→REFINED`, duration=3614ms
  - retrieval-out: `{"crag_decision": "AMBIGUOUS\u2192REFINED"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we jump to formulas, let's start with what you already know. When you see something moving — say, a car rolling down the street or a ball flying through the air — what do you think determines how much energy that motion carries?
    ```

</details>

### Turn 1

**Student:** Half m v squared.

**Tutor:**

> You've written down a mathematical expression. That formula does appear in physics, but I want to make sure you understand what it represents, not just recall symbols. 
> 
> When v doubles—say a car goes from 10 m/s to 20 m/s—what happens to the value of that expression?

_(turn_count after: 2)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `non-empty response` — response length: 267 chars

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `kinetic energy`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1522ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=1 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    You've written down a mathematical expression. That formula does appear in physics, but I want to make sure you understand what it represents, not just recall symbols. 
    
    When v doubles—say a car goes from 10 m/s to 20 m/s—what happens to the value of that expression?
    ```

</details>

---

## PA3 — Cooperative trajectory (projectile motion)

_Concept lock + turn-0 broad opener for projectile motion._

- session_id: `e2e-1c4a356426`
- result: **PASS**

### Turn 0

**Student:** How does gravity affect a ball thrown horizontally?

**Tutor:**

> Before we dive in, what do you already know about how an object can move in more than one direction at the same time?

_(turn_count after: 1)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `ends with question` — ends with a question
- ✅ `no concept reveal` — no discriminating part of 'projectile motion' present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `projectile motion`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`CORRECT`, duration=1924ms
  - retrieval-out: `{"crag_decision": "CORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we dive in, what do you already know about how an object can move in more than one direction at the same time?
    ```

</details>

---

## PB1 — Wrong attempt (force = mass)

_Common misconception — student says force = mass. Tutor must NOT confirm and must NOT reveal F=ma yet._

- session_id: `e2e-99742baea5`
- result: **PASS**

### Turn 0

**Student:** What does Newton's second law actually say?

**Tutor:**

> Before we dig into the formal statement, let's start with your intuition. When you push a shopping cart or a heavy box, what do you notice about how hard you have to push versus how fast it speeds up?

_(turn_count after: 1)_

**Assertions:**

- ✅ `no concept reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`CORRECT`, duration=1622ms
  - retrieval-out: `{"crag_decision": "CORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we dig into the formal statement, let's start with your intuition. When you push a shopping cart or a heavy box, what do you notice about how hard you have to push versus how fast it speeds up?
    ```

</details>

### Turn 1

**Student:** Is it just that force equals mass?

**Tutor:**

> You're touching on part of the picture—force and mass are definitely involved. But your statement is missing something crucial: when you push that shopping cart, the cart doesn't just sit there, it *does* something in response to your push.
> 
> What happens to the cart's motion when you apply a force to it?

_(turn_count after: 2)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `does NOT confirm 'force equals mass'` — did not confirm 'force equals mass'
- ✅ `ends with question` — ends with a question

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1788ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=1 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    You're touching on part of the picture—force and mass are definitely involved. But your statement is missing something crucial: when you push that shopping cart, the cart doesn't just sit there, it *does* something in response to your push.
    
    What happens to the cart's motion when you apply a force to it?
    ```

</details>

---

## PC1 — Three IDKs → reveal of Newton's second law

_Three consecutive 'I don't know' messages must trigger reveal via teach_node (idk_count >= IDK_REVEAL_THRESHOLD)._

- session_id: `e2e-250c62fa96`
- result: **PASS**

### Turn 0

**Student:** What does Newton's second law actually say?

**Tutor:**

> Before we explore that specific principle, let's start with what you already understand. When you see an object speed up, slow down, or change direction — like a soccer ball after you kick it — what do you think is responsible for that change in motion?

_(turn_count after: 1)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`CORRECT`, duration=2230ms
  - retrieval-out: `{"crag_decision": "CORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we explore that specific principle, let's start with what you already understand. When you see an object speed up, slow down, or change direction — like a soccer ball after you kick it — what do you think is responsible for that change in motion?
    ```

</details>

### Turn 1

**Student:** I don't know.

**Tutor:**

> When motion changes, there's always a relationship between three key ingredients: what's causing the change, how much "stuff" is being moved, and how quickly the motion is changing. Think about pushing a shopping cart versus pushing a car — both empty, both on flat ground.
> 
> What difference would you notice about how hard you'd need to push to get them moving at the same rate?

_(turn_count after: 2)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1927ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=1 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    When motion changes, there's always a relationship between three key ingredients: what's causing the change, how much "stuff" is being moved, and how quickly the motion is changing. Think about pushing a shopping cart versus pushing a car — both empty, both on flat ground.
    
    What difference would you notice about how hard you'd need to push to get them moving at the same rate?
    ```

</details>

### Turn 2

**Student:** Still no idea.

**Tutor:**

> The push you apply (the force) would need to be much stronger for the car because it has more mass than the shopping cart. There's actually a precise mathematical relationship here: the force needed is proportional to both the mass of the object and how quickly you want to change its speed.
> 
> If you wanted to make the car accelerate twice as fast, what would happen to the amount of force you'd need to apply?

_(turn_count after: 3)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1579ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=2 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    The push you apply (the force) would need to be much stronger for the car because it has more mass than the shopping cart. There's actually a precise mathematical relationship here: the force needed is proportional to both the mass of the object and how quickly you want to change its speed.
    
    If you wanted to make the car accelerate twice as fast, what would happen to the amount of force you'd need to apply?
    ```

</details>

### Turn 3

**Student:** Just tell me, I give up.

**Tutor:**

> The answer is **Newton's second law**, commonly expressed as F = ma (force equals mass times acceleration). This law tells us that the force applied to an object is equal to the object's mass multiplied by its acceleration—in other words, the same force will accelerate a lighter object more than a heavier one. In occupational therapy contexts, this principle helps us understand how much force a client needs to generate to move objects of different masses, or how assistive devices can reduce the force requirements for daily activities.
> 
> What would you like to do next?
> A) Try a clinical application question for this concept
> B) Move on to the next topic
> C) Stop here for now

_(turn_count after: 1)_

**Assertions:**

- ✅ `reveals concept` — concept "Newton's second law" revealed as expected

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1732ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=0 input_keys=['chunk_sources', 'classifier_output', 'concept_mastered', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mastery_level', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count', 'weak_topics']
  - draft (post-strip):
    ```
    The answer is **Newton's second law**, commonly expressed as F = ma (force equals mass times acceleration). This law tells us that the force applied to an object is equal to the object's mass multiplied by its acceleration—in other words, the same force will accelerate a lighter object more than a heavier one. In occupational therapy contexts, this principle helps us understand how much force a client needs to generate to move objects of different masses, or how assistive devices can reduce the force requirements for daily activities.
    
    What would you like to do next?
    A) Try a clinical applicat…
    ```

</details>

---

## PD1 — Off-topic injection (restaurant)

_Student injects a totally off-topic question. Tutor must redirect back to physics (not the previous OT 'anatomy' framing, which would be a generalizability bug)._

- session_id: `e2e-d84fd14999`
- result: **PASS**

### Turn 0

**Student:** What's the best restaurant in Buffalo?

**Tutor:**

> Ha, I appreciate the question, but I'm actually here to help you prep for your physics midterm! 🙂 
> 
> What topics are you thinking about tackling today, or is there something from class that's been giving you trouble?

_(turn_count after: 1)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `non-empty response` — response length: 215 chars
- ✅ `redirect or short` — physics-redirect phrasing present

<details><summary>Diagnostics</summary>

- **node trace:** `concept_extraction` → `retrieval`
- **retrieval:** crag_decision=`INCORRECT`, duration=1506ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`

</details>

---

## PG1 — Vague opener ('tell me about physics')

_Pure-vague opener — manager rejects the concept and rapport narrows. Must NOT crash, must NOT lock onto a wrong concept._

- session_id: `e2e-92c2ddb2f8`
- result: **PASS**

### Turn 0

**Student:** Tell me about physics.

**Tutor:**

> I'd love to help you prep for your midterm! Physics covers a lot of ground, so let me ask: are you looking to work through a specific topic that's been tricky, or would you rather explore something that caught your curiosity from the lectures?

_(turn_count after: 1)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `non-empty response` — response length: 243 chars
- ✅ `rapport-style reply` — rapport-style: 243 chars, ends with '?', cue present

<details><summary>Diagnostics</summary>

- **node trace:** `concept_extraction` → `retrieval`
- **retrieval:** crag_decision=`AMBIGUOUS`, duration=3377ms
  - retrieval-out: `{"crag_decision": "AMBIGUOUS"}`

</details>

---

## PI1 — First-try mastery menu (kinetic energy)

_Student names the concept correctly on attempt 1 → mastery menu (A/B/C) fires._

- session_id: `e2e-7a2654ecc1`
- result: **PASS**

### Turn 0

**Student:** What's the formula that relates an object's mass and speed to its energy of motion?

**Tutor:**

> Before we dive into formulas, let's start with your intuition. When you think about a moving object—say, a baseball flying through the air or a truck rolling down the street—what properties of that object do you think would affect how much "oomph" it has in its motion?

_(turn_count after: 1)_

**Assertions:**

- ✅ `no reveal at turn 0` — no discriminating part of 'kinetic energy' present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `kinetic energy`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`AMBIGUOUS`, duration=3508ms
  - retrieval-out: `{"crag_decision": "AMBIGUOUS"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we dive into formulas, let's start with your intuition. When you think about a moving object—say, a baseball flying through the air or a truck rolling down the street—what properties of that object do you think would affect how much "oomph" it has in its motion?
    ```

</details>

### Turn 1

**Student:** It's kinetic energy, KE = 1/2 m v squared.

**Tutor:**

> That's correct — you've identified the kinetic energy.
> 
> What would you like to do next?
> A) Try a clinical application question for this concept
> B) Move on to the next topic
> C) Stop here for now

_(turn_count after: 1)_

**Assertions:**

- ✅ `mastery menu present` — A/B/C menu present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `kinetic energy`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1742ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=0 input_keys=['chunk_sources', 'classifier_output', 'concept_mastered', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mastery_level', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    That's correct — you've identified the kinetic energy.
    
    What would you like to do next?
    A) Try a clinical application question for this concept
    B) Move on to the next topic
    C) Stop here for now
    ```

</details>

---

## PL2 — Wrong mechanism (force = velocity)

_Student claims force = velocity (a common conflation). Tutor scaffolds without revealing F=ma. Verifies multi-turn concept-lock preservation on physics._

- session_id: `e2e-4fa1f4bf58`
- result: **PASS**

### Turn 0

**Student:** What does Newton's second law actually say?

**Tutor:**

> Before we dive in, let's start with what you already know. When you see an object speed up, slow down, or change direction — what do you think is responsible for that change in motion?

_(turn_count after: 1)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`CORRECT`, duration=1621ms
  - retrieval-out: `{"crag_decision": "CORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we dive in, let's start with what you already know. When you see an object speed up, slow down, or change direction — what do you think is responsible for that change in motion?
    ```

</details>

### Turn 1

**Student:** Doesn't force just equal velocity?

**Tutor:**

> That's not quite right—velocity describes motion, but it doesn't capture what force does. Think about a hockey puck sliding across ice at constant velocity: once it's moving, it keeps going without any force (ignoring friction). Force becomes important when something about the motion *changes*.
> 
> What happens to an object's velocity when a force acts on it?

_(turn_count after: 2)_

**Assertions:**

- ✅ `not fallback_scaffold` — not the fallback message
- ✅ `no reveal` — no discriminating part of "Newton's second law" present
- ✅ `ends with question` — ends with a question
- ✅ `≤ 2 question marks` — 1 '?' in response

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1947ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=1 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    That's not quite right—velocity describes motion, but it doesn't capture what force does. Think about a hockey puck sliding across ice at constant velocity: once it's moving, it keeps going without any force (ignoring friction). Force becomes important when something about the motion *changes*.
    
    What happens to an object's velocity when a force acts on it?
    ```

</details>

---

## PW1 — IDK reveal seeds weak_topics

_Concept revealed via IDK ladder must land in weak_topics so the dashboard can prioritize it next session._

- session_id: `e2e-6b9930649c`
- result: **PASS**

### Turn 0

**Student:** What does Newton's second law actually say?

**Tutor:**

> Before we work through this together, let's start with what you've already observed in the world around you. When you push a shopping cart or kick a soccer ball, what do you notice about how the object's motion changes? What factors seem to matter?

_(turn_count after: 1)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `generation` → `dean`
- **retrieval:** crag_decision=`CORRECT`, duration=1833ms
  - retrieval-out: `{"crag_decision": "CORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=? input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice']
  - draft (post-strip):
    ```
    Before we work through this together, let's start with what you've already observed in the world around you. When you push a shopping cart or kick a soccer ball, what do you notice about how the object's motion changes? What factors seem to matter?
    ```

</details>

### Turn 1

**Student:** I don't know.

**Tutor:**

> When objects speed up, slow down, or change direction, something must be causing that change. In physics, we study how forces relate to changes in motion—this is the heart of dynamics. 
> 
> What do you think determines *how much* an object's motion changes when you apply a force to it?

_(turn_count after: 2)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `concept_extraction` → `retrieval` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1833ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=1 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    When objects speed up, slow down, or change direction, something must be causing that change. In physics, we study how forces relate to changes in motion—this is the heart of dynamics. 
    
    What do you think determines *how much* an object's motion changes when you apply a force to it?
    ```

</details>

### Turn 2

**Student:** Still no idea.

**Tutor:**

> There's a precise mathematical relationship that connects three quantities: the force you apply, the mass of the object, and how quickly its velocity changes (acceleration). Heavier objects need more force to achieve the same acceleration, while lighter objects need less force.
> 
> If you double the force on an object while keeping its mass constant, what happens to its acceleration?

_(turn_count after: 3)_

**Assertions:**

- ✅ `no reveal` — no discriminating part of "Newton's second law" present

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1440ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=2 input_keys=['chunk_sources', 'classifier_output', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count']
  - draft (post-strip):
    ```
    There's a precise mathematical relationship that connects three quantities: the force you apply, the mass of the object, and how quickly its velocity changes (acceleration). Heavier objects need more force to achieve the same acceleration, while lighter objects need less force.
    
    If you double the force on an object while keeping its mass constant, what happens to its acceleration?
    ```

</details>

### Turn 3

**Student:** Just tell me.

**Tutor:**

> The answer is **Newton's second law**, often written as F = ma (force equals mass times acceleration). This law tells us that the net force acting on an object is equal to the object's mass multiplied by its acceleration—in other words, heavier objects require more force to accelerate at the same rate as lighter ones, and greater acceleration demands proportionally greater force. In occupational therapy, this principle helps us understand the forces clients generate during movement, the resistance needed in therapeutic activities, and how joint loading changes with different masses and movement speeds.
> 
> What would you like to do next?  
> A) Try a clinical application question for this concept  
> B) Move on to the next topic  
> C) Stop here for now

_(turn_count after: 1)_

**Assertions:**

- ✅ `reveals concept` — concept "Newton's second law" revealed as expected

<details><summary>Diagnostics</summary>

- **manager_agent → concept:** `Newton's second law`
- **node trace:** `retrieval` → `concept_extraction` → `classifier` → `dean`
- **retrieval:** crag_decision=`INCORRECT`, duration=1574ms
  - retrieval-out: `{"crag_decision": "INCORRECT"}`
- **dean call #0**: passed=✅ revisions_after=0 turn=0 input_keys=['chunk_sources', 'classifier_output', 'concept_mastered', 'crag_decision', 'current_concept', 'dean_passed', 'dean_revision_instruction', 'dean_revisions', 'domain', 'draft_response', 'draft_source_node', 'idk_count', 'mastery_level', 'mode', 'session_id', 'student_attempted', 'student_phase', 'topic_choice', 'turn_count', 'weak_topics']
  - draft (post-strip):
    ```
    The answer is **Newton's second law**, often written as F = ma (force equals mass times acceleration). This law tells us that the net force acting on an object is equal to the object's mass multiplied by its acceleration—in other words, heavier objects require more force to accelerate at the same rate as lighter ones, and greater acceleration demands proportionally greater force. In occupational therapy, this principle helps us understand the forces clients generate during movement, the resistance needed in therapeutic activities, and how joint loading changes with different masses and movemen…
    ```

</details>

### Post-scenario checks

- ✅ `Newton's second law added to weak_topics after IDK reveal` — "Newton's second law" present in weak_topics=["Newton's second law"]
  - **weak_topics in session:** `["Newton's second law"]`
  - **current_concept in session:** `"Newton's second law"`

---
