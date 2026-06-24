# Ownership Plan — Understanding This Project Well Enough to Defend It

> **Status: experimental / living document.**
> This is a study plan, not project documentation. Its goal: take a codebase I
> didn't write line-by-line and reach the point where I can re-derive every
> metric and defend every design decision from scratch — including under a hard
> technical interview.

This file is written as a **tutor-assisted plan**: Claude acts as a tutor that
writes tutor docs, quizzes me hard, then I rewrite each doc in my own words.
The doc that survives is one *I* wrote from knowledge *I* recalled under
pressure. That is what real ownership means.

---

## The principle

**Ownership ≠ authorship.** Plenty of senior engineers own code they didn't
originally write. Ownership means:

> I can defend every decision and re-derive every number from scratch.

The vehicle for getting there is a set of writeups. The writeups are not the
point — the *understanding they force* is.

## The test for "do I actually own this?"

For each component, can I answer these four **without notes**:

1. **Intuition** — what problem does it solve, in one sentence to a non-expert?
2. **Math** — derive the formula on a whiteboard. *This is where most people fold.*
3. **Failure modes** — when does it break / give a wrong number?
4. **Why this, not the alternative** — what did I reject, and why?

If I can't do #2 and #4, I don't own it yet. Hard interviews live almost
entirely in #2 and #4.

---

## The writeup structure to build

One file per component, each following the same 4-part template above
(intuition → math → failure modes → why-this-not-that).

```
writeups/
  00-ownership-plan.md   ← this file
  01-retrieval.md        ← embeddings, cosine sim, dense vs sparse, reranking
  02-agent-graph.md      ← LangGraph, state machines, why a graph not a chain
  03-ragas-math.md       ← the 4 RAGAS metric formulas (HIGHEST leverage)
  04-llm-judge.md        ← self-consistency, variance, Cohen's κ
  05-trajectory.md       ← path scoring, the actual differentiator of this project
  06-regression-gate.md  ← thresholds, statistical validity
  07-design-decisions.md ← cross-family judging, the "why" log
```

---

## Where the real math is (and what interviews probe)

This project is unusually math-rich for a "RAG project" because the **eval
metrics have real formulas** — most candidates only know cosine similarity.

| Component | The actual math | Brutal interview question it defends against |
|-----------|-----------------|---------------------------------------------|
| Retrieval | `cosine sim = (A·B) / (‖A‖‖B‖)` | "Why cosine not Euclidean? What does the dot product mean geometrically?" |
| RAGAS **faithfulness** | `supported_claims / total_claims` | "How do you decompose an answer into 'claims'? What's the failure mode?" |
| RAGAS **answer_relevancy** | `mean cos-sim(question, N reverse-generated questions)` | "Why generate questions *backwards* from the answer? Why is that clever?" |
| RAGAS **context_precision** | weighted Average Precision over ranked contexts | "This is just MAP from IR — connect it to ranking. Why does order matter?" |
| RAGAS **context_recall** | `attributable_GT_sentences / total_GT_sentences` | "Why was mine 0.0? Recall vs precision tradeoff here?" |
| Judge | self-consistency: median of N samples; variance = std as confidence | "Why median not mean? What does high variance tell you?" |
| Calibration | `Cohen's κ = (p_o − p_e) / (1 − p_e)` | "Why κ not raw accuracy? What does it correct for?" |

**Key cross-link:** RAGAS `context_precision` *is* Mean Average Precision from
information retrieval — which connects straight to the **reranking** subtlety in
retrieval. Understanding one unlocks the other.

---

## The method (active recall, not passive reading)

Reading explanations does not create ownership — I'd forget it in a week.
**Active recall under pressure** does. So, per document:

1. **Tutor writes the tutor version** — intuition + full math derivation +
   failure modes, grounded in the actual code.
2. **I read it once.**
3. **Tutor quizzes me hard** — plays the brutal interviewer; I answer in my own
   words.
4. **Tutor corrects the gaps; I rewrite the doc in my own words.** ← the
   ownership step. The committed version is the one I wrote.

---

## Order of attack

Start with **`03-ragas-math.md`** — highest leverage because:
- it's the densest math,
- it's the most unique part of this project,
- `context_precision = MAP` ties back into retrieval/reranking, so it unlocks
  two docs at once.

Then roughly: `01-retrieval` → `04-llm-judge` → `05-trajectory` →
`02-agent-graph` → `06-regression-gate` → `07-design-decisions`.

---

## Progress tracker

- [ ] 01-retrieval
- [ ] 02-agent-graph
- [ ] 03-ragas-math  ← **start here**
- [ ] 04-llm-judge
- [ ] 05-trajectory
- [ ] 06-regression-gate
- [ ] 07-design-decisions

> Each box gets checked only after I've passed the quiz **and** rewritten the
> doc in my own words. Reading the tutor version doesn't count.
