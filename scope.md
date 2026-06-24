Pasted markdown(1).md
File
review this design :

Pasted markdown(2).md
File

v0.2 is much better aligned with what recruiters are actually searching for today.

If I compare the three versions:
Version	Technical Depth	Recruiter Signal	Risk
Pure RAG Eval Harness	9/10	6.5/10	Looks niche
Agentic Eval Harness (v0.2)	9.5/10	9/10	Higher scope
Generic Agentic RAG App	6/10	9/10	Commodity

v0.2 sits in a nice spot because it has both:

    trendy surface area (LangGraph, Agents, Tool Calling, Observability)

    deep engineering substance (evaluation, regression testing, calibration)

That combination is rare.

My main concern now is different from before:

You're building an evaluation platform for a problem you haven't experienced yet.

The strongest evaluation systems are usually born from pain:

    "Our agent kept breaking and we couldn't tell why."

Then observability and regression testing become obviously necessary.

Here the danger is:

Agent
↓
Trajectory scorer
↓
Custom judge
↓
Observability
↓
Regression tracker
↓
FastAPI
↓
CI

before you've even observed a real failure.

I would reverse the order.

Build:

    Agent

    Tracing

    Golden set

    Break the agent intentionally

    Evaluation

    CI

The key artifact isn't the architecture diagram.

It's:

    "Changing planner prompt X caused tool-selection accuracy to drop from 91% → 68% on multi-hop questions, and the regression gate caught it."

That's the story.

A few things still feel slightly buzzwordy:
Excessive Agency

This is academically interesting, but your agent is:

arXiv corpus
vector search
answer questions

There is no real agency.

Interviewers may ask:

    What action could your agent actually take?

And the answer is basically:

    retrieve documents

So "excessive agency" feels imported from OWASP rather than emerging naturally from the system.

I'd probably replace it with:

    planning quality

    tool selection correctness

    retrieval strategy quality

unless you actually give the agent external actions.
OpenAI + Anthropic

Having both in the stack is good.

But don't force dual-provider support unless you genuinely benchmark them.

Otherwise it becomes resume decoration.
Golden trajectories

This is where I think the real novelty is.

Most projects evaluate:

question
answer

Very few evaluate:

question
trajectory
answer

If you can genuinely score:

    wrong tool

    unnecessary tool

    missed retrieval

    inefficient path

that's substantially more interesting than another RAG benchmark.
What recruiters will notice

Resume bullet:

Built a continuous evaluation and observability platform for LangGraph-based AI agents, combining trajectory scoring, LLM-as-judge evaluation, regression testing, tracing, and CI/CD deployment gates.

gets attention.

Resume bullet:

Built a RAG chatbot over arXiv papers.

does not.

If your goal is maximizing interview signal, I'd rate v0.2 around 8.5–9/10.

The only project I think would score higher right now would be something like:

Agentic AI Platform
├── LangGraph agents
├── MCP tools
├── Evaluation harness
├── Observability
├── Human feedback loops
└── Deployment infrastructure

because that captures almost every keyword cluster currently appearing in GenAI Engineer and AI Platform Engineer postings.

But among evaluation-focused projects, v0.2 is much closer to something a hiring manager would believe was built by an ML engineer rather than a student following a tutorial.

ChatGPT is mostly right, and the two corrections worth taking are corrections I should have caught. Let me separate the keep from the discard, because one of its suggestions is good enough to restructure around and one would actively hurt you.

**Take it — build order is the real fix.** This is the strongest point made across this entire thread, by either of us. An eval platform built before observing a failure is solving a problem on faith, and an interviewer can hear the difference in fifteen seconds. The reordered sequence — agent → tracing → golden set → **break it on purpose** → eval → CI — isn't just better engineering, it's better *epistemics*: you build the regression gate because you watched a regression happen, not because a design doc told you to. That inverts the v0.2 phase order and I should update the doc to match. The artifact ChatGPT names — "planner prompt X dropped tool-selection 91%→68% on multi-hop, gate caught it" — is exactly story #4 in my own doc, and ChatGPT is correctly pointing out that the build order should be organized *around producing that artifact* rather than treating it as a phase-9 afterthought.

**Take it — excessive agency is genuinely overclaimed, and this one's on me.** ChatGPT is right and the catch is sharp. An agent whose only tool is vector-search over arXiv has no irreversible actions to take, so an "excessive-agency" rubric is a solution looking for a problem — imported from OWASP to look sophisticated, not emergent from the system. The interview question "what action could your agent actually take?" would expose it instantly, and that's the worst kind of resume line: one that *invites* a question you can't answer well. Its proposed replacements — planning quality, tool-selection correctness, retrieval-strategy quality — actually arise from the system you're building. I'll cut excessive agency and replace it. (The only way to keep it honestly would be to give the agent a real external action — a web-fetch tool, a write action — and then it becomes legitimate. But that's scope creep for a keyword; not worth it.)

**Take it, with discipline — dual-provider.** Correct that OpenAI+Anthropic-as-decoration is transparent. But there's a clean way to make it real instead of cutting it: your *judge* uses a different provider than the *agent under test*. That's not decoration — using the same model family to judge its own outputs causes the well-known self-preference bias, which your own §4.3 already worries about. So agent-on-OpenAI, judge-on-Anthropic (or vice versa) is a *defensible engineering reason* for both being in the stack, and it strengthens the judge-design story. That's the version to keep; "I support both providers" with no benchmark is the version to drop.

Now where I'll push back on ChatGPT, because deferring to it here would set you back:

**The "Agentic AI Platform" with everything is a trap, and it's the same trap from two rounds ago wearing a higher score.** ChatGPT rates it as the only thing that'd score higher, then lists six subsystems including human-feedback loops and deployment infrastructure. That's not a project; it's a team's roadmap. You have a target of August and a documented pattern of choosing scope over shipping. A platform that's 50% built reads as *worse* than a harness that's 100% done with one real regression caught — because the platform has no finished story and the harness has story #4. ChatGPT even undercuts its own suggestion in its last line: v0.2 is "something a hiring manager would believe was built by an ML engineer rather than a student following a tutorial." The everything-platform is the tutorial-follower's instinct (breadth as proxy for competence). The narrow, finished, failure-driven harness is the engineer's. Don't trade the thing that signals seniority for the thing that signals keyword anxiety.

**One thing ChatGPT slightly overstates: "no real agency" → therefore trajectory eval is the only novelty.** The trajectory eval *is* the novelty and ChatGPT is right to center it. But even a retrieve-only agent has genuinely scorable path decisions — did it decompose a multi-hop question or one-shot it, did it retrieve twice when once suffced, did it re-retrieve after a bad observation or give up. Those are real, gradeable, and most projects don't touch them. So the system isn't as thin as "it just retrieves" implies — the *answers* are commodity, the *paths to them* are not. That's the whole bet, and it survives.

Net: ChatGPT moved the doc forward on two real axes (build order, excessive agency) and I'm taking both. The scope-up suggestion is the one place to hold the line hard.

Want me to ship v0.3 with: build order reversed around the break-it-first artifact, excessive-agency cut and replaced with planning-quality + tool-selection + retrieval-strategy, dual-provider reframed as agent/judge separation, and the scope explicitly capped against the platform temptation? That's a tight, honest revision and then you build rather than redesign again — this is the last design pass before the work is just work.

I agree with most of that critique.

The biggest thing I noticed reading both v0.1 and v0.2 is that the proposal became increasingly optimized for sounding like something a staff engineer would say, rather than maximizing the probability that a single person ships it before recruiting season.

The strongest sentence in the entire discussion is probably:

    "Build the regression gate because you watched a regression happen."

That's the difference between a project and a framework.

If I were hiring, these two stories are not equal:

Story A

    Built LangGraph agent

    Built evaluator

    Built tracing

    Intentionally changed retrieval strategy

    Multi-hop success rate collapsed

    Eval suite detected it

    CI blocked merge

    Investigated trace

    Fixed issue

Story B

    Built evaluation platform

    Supports trajectory scoring

    Supports judge calibration

    Supports observability

    Supports regression tracking

    Supports CI

Story A sounds real. Story B sounds designed.

The excessive-agency point is also correct. When I read that section, my first reaction was:

    "What dangerous thing can this agent actually do?"

If the answer is "query a vector database," then OWASP-style excessive-agency evaluation feels detached from the system being evaluated. Planning quality, retrieval strategy quality, and tool-selection quality are much harder to attack because they're obviously relevant.

The dual-provider reframing is also stronger than the original. A judge evaluating outputs from a different model family is a legitimate design choice. Saying "supports OpenAI and Anthropic" is resume padding unless you're actually comparing them.

The place where I'd push even harder than Claude's response is on the golden dataset.

I think the golden dataset is still the real moat.

Anybody can build:

    LangGraph agent

    LangSmith tracing

    FastAPI service

    GitHub Action

in a weekend.

Almost nobody will spend days creating:

    adversarial multi-hop cases

    retrieval-ground-truth annotations

    trajectory annotations

    unanswerable questions

That's where originality comes from.

If this were my project, I'd allocate effort roughly like:

    40% golden dataset

    20% trajectory scoring

    15% tracing

    10% regression analysis

    10% agent

    5% service + CI

Most people would invert those numbers. That's why most portfolio projects look the same.

My only remaining concern is that even v0.2 may still be trying to solve three problems:

    Agent evaluation

    Agent observability

    Regression infrastructure

You can make a credible project from any one of those.

The sweet spot is where they reinforce each other rather than becoming separate deliverables.

A useful litmus test:

    If I removed FastAPI, Docker, GitHub Actions, and W&B entirely, would the project still be impressive?

If the answer is yes, you're focused on the right thing.

For v0.2, I think the answer is mostly yes now, which is a good sign.

At this point I'd stop redesigning and start building. The next meaningful design insight is unlikely to come from another architecture document. It's going to come from the first time your agent does something dumb, your trace explains why, and your evaluator either catches it or fails to catch it. That's where the real project begins.

Pasted markdown(3).md
File

This is the first version where I stop thinking about redesigns and start thinking about implementation.

The biggest improvement is not any individual section—it's that the document now has a center of gravity.

In v0.1 the center was "evaluation."

In v0.2 the center started drifting toward "agent platform."

In v0.3 the center is clearly:

    Build an agent → observe failure → create a dataset around that failure surface → build evaluation that catches it → enforce regression prevention.

That's a coherent story.

A few things stand out as particularly strong:

1. Section 9 (Build Order) is now the strongest part of the document.

Previously the project felt architecture-driven.

Now it feels evidence-driven.

This sequence:

agent
→ tracing
→ observe failure
→ build golden set
→ build evaluator
→ build regression tracker

is much harder to criticize because every layer justifies the next one.

If an interviewer asks:

    "Why did you build trajectory scoring?"

you have a concrete answer:

    "Because I observed the agent repeatedly one-shotting multi-hop questions."

That is far more convincing than:

    "Because agent trajectory evaluation is important."

2. The excessive-agency removal was absolutely correct.

The replacement dimensions:

    planning quality

    tool selection

    retrieval strategy

    efficiency

    goal completion

all emerge naturally from the system.

Nothing feels imported anymore.

3. The litmus test in §11 is excellent.

Honestly I'd move it earlier.

Something like:

Core Principle:
If FastAPI, Docker, GitHub Actions,
and W&B disappeared tomorrow,
the project should still be impressive.

That sentence alone protects you from months of wasted effort.

4. The effort allocation is realistic.

Most people claim:

40% evaluation
40% infra
20% dataset

and then end up with:

70% infra
20% agent
10% evaluation

Your allocation now explicitly fights that failure mode.

That's good.

The one remaining thing I'd tighten is the trajectory scoring section.

Right now:

Planning quality
Tool selection
Retrieval strategy
Efficiency
Goal completion

are good categories.

But I still don't know exactly how they'll be scored.

That's where interviewers may probe.

For example:

Planning quality

What is the metric?

gold trajectory overlap?
judge score?
edit distance?

Retrieval strategy

What is success?

retrieved expected chunks?
retrieved all hops?
retrieval recall?

Efficiency

actual_steps / optimal_steps ?

You don't need full formulas yet, but I'd eventually want a one-page "trajectory scoring specification."

That document will likely end up being more valuable than another architecture diagram.

One subtle thing I noticed:

The document now contains two genuinely uncommon artifacts:

    Retrieval-ground-truth annotations

    Trajectory-ground-truth annotations

Those are much rarer than:

    LangGraph

    LangSmith

    FastAPI

    RAGAS

If I were prioritizing implementation, I'd protect those first.

If time becomes constrained, I'd rather see:

100-item dataset
great trajectory annotations
basic FastAPI

than:

beautiful API
Docker
GitHub Actions
20 weak examples

because the former is what creates defensible differentiation.

My assessment now:

Technical depth: 9/10

Recruiter keyword coverage: 9/10

Likelihood of actually shipping before recruiting season: 8.5/10

Risk of scope creep: down from ~8/10 to ~4/10

Resume value if completed: very high

The most important sentence in the entire document is no longer any architecture section.

It's this one:

    "The next meaningful insight will not come from another architecture document—it will come from the first time the agent does something dumb."

That is probably true. At this point, I'd freeze the design and start Step 1.


Continuous Evaluation & Observability Harness for Agentic RAG Systems
Design & Scope Specification — v0.3 (final pre-build)

Author: Ramjan Khandelwal · github.com/ravenZ3 · ravenZ3.github.io
Status: Design locked — next step is build, not revision · Target: A single-person-shippable evaluation + observability harness for a LangGraph agent, finished before recruiting season.

    What changed v0.2 → v0.3 (and why this is the last design pass). Four corrections, all adopted: (1) build order reversed — build the regression gate because you watched a regression happen, not on faith; (2) excessive-agency cut — replaced with planning-quality, tool-selection-correctness, and retrieval-strategy-quality, which actually emerge from a retrieve-only agent; (3) dual-provider reframed as agent/judge model-family separation (a real bias-control reason, not resume padding); (4) golden dataset promoted to the explicit center of gravity — it is the moat, and effort is allocated accordingly. Plus a governing litmus test (§11) the whole project must keep passing. After this, the next meaningful insight comes from the agent doing something dumb and the harness catching it — not from another document.

0. One-line

A continuous evaluation and observability harness that treats agentic-RAG quality as a regression-tested engineering artifact — scoring not just was the answer faithful but did the agent plan well, retrieve the right things, choose the right tools, and take an efficient path — with reference-free metrics, a calibrated cross-family LLM judge, trace-level observability, a hand-curated golden set as its moat, and a CI gate that blocks merges on regression.
1. The problem this exists to solve

Agentic RAG fails in more ways than linear RAG, and worse, it fails invisibly across steps. An agent can plan badly, retrieve once when it needed twice, ignore what it retrieved, choose a wrong tool, or wander to a correct answer by luck in six steps when two would do. A single end-of-pipeline faithfulness score is blind to all of it.

Two industry facts make this the right thing to build in 2026. First, eval is the single highest-signal skill in LLM hiring — the named interview test is "walk me through an evaluation you designed," and the answer quality is the screen. Second, agent evaluation specifically is an open, hard problem few candidates can speak to.

The thesis defended in the interview: the hard part of eval is not computing metrics — it is making metrics trustworthy enough to block a deploy on. Agents raise the bar, because the unit being judged is a trajectory: noisier, longer, harder to ground than a single answer.

The honest origin story this project is built to earn. The strongest eval systems are born from pain — "our agent kept breaking and we couldn't tell why" — and then observability and regression testing become obviously necessary. v0.3 is sequenced so that story is true rather than retrofitted: the agent and tracing come first, a real failure is observed, and only then is the evaluator and gate built to catch it. The key artifact is not the architecture diagram. It is one sentence: "Changing the planner prompt dropped tool-selection accuracy from 91% → 68% on multi-hop questions, and the regression gate caught it before merge." Everything in this doc serves the production of that sentence.
2. Scope boundary (what is and isn't the project)
Layer	Role	Investment
Agentic RAG pipeline	The subject under test. LangGraph agent: plan → retrieve → (re-retrieve / tool) → observe → synthesize, over an arXiv ML corpus in a vector DB.	Thin-but-real. ≥2 tool-use turns + explicit state, or trajectory eval is meaningless. Deliberately allowed to be mediocre.
Tracing / observability	Per-step trace capture from the first run (LangSmith / Phoenix); failures surface with the trajectory that produced them.	Build early, not late. It's how you observe the failure that justifies everything downstream.
Golden dataset	Hand-curated answer triples + trajectory cases, adversarial multi-hop, unanswerables, retrieval ground truth.	The moat. The single largest effort allocation (§10).
Eval engine	RAGAS (answer-level) + trajectory scorer (agent-level) + calibrated cross-family judge.	Deep. The core.
Regression tracker	W&B runs, baseline diff, per-category + per-step flagging.	Medium. Earns "production-grade," but small relative to the dataset.
API + CI	FastAPI service; GitHub Action merge gate.	Small. ~5% of effort. Plumbing, not the point (see litmus test, §11).

Explicit non-goals — and a hard scope cap. No better agent, retriever, or SOTA chase. No arbitrary-framework support. And explicitly not an "agentic AI platform" with human-feedback loops, deployment infra, and six subsystems — that is a team's roadmap, not a single person's pre-recruiting project, and a 50%-built platform reads worse than a 100%-finished harness with one real regression caught. Story A ("I broke it, the eval caught it, I fixed it") beats Story B ("it supports trajectory scoring, judge calibration, observability, regression tracking, CI") — A sounds real, B sounds designed. Build for A.
3. System architecture

                         ┌─────────────────────────────────────────────┐
                         │              GitHub Action (PR gate)          │
                         │   on: pull_request → POST /run → assert pass  │
                         └───────────────────────┬─────────────────────┘
                                                 │
                    ┌────────────────────────────▼────────────────────────────┐
                    │                      FastAPI service                      │
                    │   /evaluate            (sync, answer → answer scores)    │
                    │   /evaluate-trajectory (sync, trace → trajectory scores) │
                    │   /run                 (async, full suite → run_id)      │
                    │   /runs/{id}           (poll status + results + traces)  │
                    └───┬──────────────────────────────────────────────┬──────┘
                        │                                               │
          ┌─────────────▼──────────────────────────┐     ┌─────────────▼─────────────┐
          │              Eval Engine                 │     │     Regression Tracker     │
          │  ┌────────────────────────────────────┐ │ ──▶ │  W&B: metrics, prompt_ver, │
          │  │ RAGAS (answer-level)               │ │     │  model_ver, agent_ver,     │
          │  │ faithfulness/AR/ctx-prec/ctx-recall│ │     │  retr_cfg, golden_hash,    │
          │  ├────────────────────────────────────┤ │     │  variance, cost/latency    │
          │  │ Trajectory scorer (agent-level)    │ │     │  ── baseline diff engine ──│
          │  │ planning-quality / tool-selection /│ │     │  per-category + per-step Δ │
          │  │ retrieval-strategy / step-efficiency│ │     │  Δ > threshold → REGRESSION│
          │  │ / goal-completion                  │ │     └────────────┬──────────────┘
          │  ├────────────────────────────────────┤ │                  │ links
          │  │ Cross-family LLM judge             │ │     ┌────────────▼─────────────┐
          │  │ safety / tone / plausible-halluc.  │ │     │   Tracing / Observability │
          │  │ (judge model ≠ agent model family) │ │ ◀── │  LangSmith / Phoenix      │
          │  └────────────────────────────────────┘ │     │  per-step spans, tool I/O,│
          └─────────────┬───────────────┬───────────┘     │  latency, token cost,     │
                        │ reads         │ consumes traces │  failure → trace link     │
          ┌─────────────▼─────────┐     │                 └────────────┬──────────────┘
          │   Golden Dataset       │     │                              │ emits spans
          │  answer-level triples  │◀────┘     ┌───────────────────────▼──────────────┐
          │  + trajectory cases    │           │   Agentic RAG pipeline (under test)    │
          │  adversarial/unanswer. │ ─────────▶│  LangGraph: [plan] → [retrieve] →      │
          │  versioned, hashed     │  queries  │  [re-retrieve/tool] → [observe] →      │
          └────────────────────────┘           │  [synthesize] · explicit state         │
                                               └────────────────────────────────────────┘

The decoupling seam holds: the eval engine never imports the agent. It consumes two contracts — an answer record (question, answer, contexts, ground_truth) for RAGAS, and a trajectory record (question, steps[], final_answer) where each step is (thought, tool_called, tool_args, tool_result, observation). Framework-agnostic on purpose, so the same harness can grade a different agent later.
4. The eval engine (the actual project)
4.1 Three scoring tracks, deliberately

Track A — RAGAS (answer-level, reference-free). Faithfulness, answer relevancy, context precision, context recall. The load-bearing quantitative metrics and a primary CI gate.

Track B — Trajectory scorer (agent-level). The reason this is an agentic eval, and where the rare signal lives. Scores the path, not just the destination. Dimensions chosen because they obviously emerge from a retrieve-only agent — they're hard to attack as imported or detached:

    Planning quality — did the agent decompose a multi-hop question into the right sub-goals, or one-shot it?

    Tool-selection correctness — at each tool call, was it the right tool/action for that sub-goal? (Graded against the golden trajectory where one exists; judge-scored otherwise.)

    Retrieval-strategy quality — did it retrieve the right things, in the right number of passes? Did it re-retrieve after a bad observation, or give up / confabulate the missing hop?

    Step efficiency — reasonable number of steps vs. wandering, measured against the golden trajectory's step count.

    Goal completion — fully satisfied the question, partially, or confidently stopped short?

    Why not "excessive agency" (OWASP LLM06)? It was in v0.2 and is cut. An agent whose only tool is vector-search over arXiv has no irreversible action to take — the honest answer to "what dangerous thing can it do?" is "query a database." An excessive-agency rubric on that system is imported to sound sophisticated, and invites an interview question with no good answer. The dimensions above are obviously relevant to the system as built, which is the whole point. (If the agent were later given a real external action — web-fetch, a write — excessive-agency becomes legitimate. That's a deliberate future scope decision, not a keyword to claim now.)

Track C — Cross-family LLM judge. Handles the squishy semantic dimensions the trajectory scorer can't do programmatically: safety, tone, plausible-hallucination. The judge runs on a different model family than the agent under test (e.g. agent on OpenAI, judge on Anthropic). This is the dual-provider justification — not "supports both," but a real bias control: a judge from the same family as the generator exhibits self-preference bias, which §4.3 explicitly worries about. Two providers in the stack is therefore an engineering decision with a defensible reason, not decoration.
4.2 Why RAGAS alone is insufficient — and what the judge catches

    Faithfulness checks claim-context entailment, not harm. An answer can be faithful to a passage and still unsafe (the necessary caveat lived in an un-retrieved chunk). RAGAS scores 1.0; the safety rubric scores it down.

    Answer-relevancy rewards topical overlap, not tone. A relevant answer in a hostile/overconfident register is a product failure RAGAS is blind to; the tone rubric catches it.

    Faithfulness misses plausible hallucination. RAGAS claim-decomposition can pass a fabricated-but-on-topic statement under broad context; the hallucination rubric runs a stricter independent atomic-claim pass.

The framing for the interview: RAGAS answers is the answer grounded and relevant?; the trajectory scorer answers did the agent get there competently?; the judge answers is it safe, well-toned, and non-fabricated? Three orthogonal lenses, and the demo that lands is one concrete case where they disagree.
4.3 Judge design — the parts that are actually hard

A naive LLM judge is non-deterministic, position-biased, sycophantic, miscalibrated — and trajectories make it worse (longer inputs, more drift). The design names these as hard:

    Rubric, not gestalt. Each dimension gets an explicit 1–5 anchored rubric, scored independently.

    Structured output + parse-or-fail. Strict JSON (score, reasoning, failing_step_index?, failing_claims[]). Unparseable output is a failed eval, not a silent zero.

    Cross-family by design. Judge model family ≠ agent model family (§4.1 Track C), to control self-preference bias.

    Calibration against my own labels. Hand-label a ~20-item slice (incl. trajectory cases), report judge–human agreement (Cohen's κ). If the judge can't agree with me on 20 items, it's broken and the README says so rather than reporting numbers I don't trust.

    Variance disclosure. Temperature 0; randomized position on pairwise dimensions; self-consistency (n=3, median) on unstable dimensions. Measure and report run-to-run variance — an eval that won't disclose its own noise floor has no business gating a deploy.

4.4 Golden dataset — the moat (see §10 for why it gets 40% of effort)

This is where originality comes from. Anyone can wire a LangGraph agent, LangSmith tracing, a FastAPI service, and a GitHub Action in a weekend. Almost nobody will spend days building adversarial multi-hop cases, retrieval-ground-truth annotations, trajectory annotations, and genuine unanswerables. That asymmetry is the entire competitive advantage of this project.

50–100 hand-curated items over the arXiv ML corpus, hybrid-built (LLM drafts from real retrieved chunks → heavy manual curation + adversarial cases), with a trajectory layer:

- id: gold_047
  question: "..."
  ground_truth: "..."
  contexts: ["chunk_id_..."]          # passages that SHOULD be retrieved
  golden_trajectory:                  # the competent path, where one exists
    - {goal: "decompose question", expected_tool: null}
    - {goal: "retrieve paper A",   expected_tool: "vector_search"}
    - {goal: "retrieve paper B",   expected_tool: "vector_search"}
    - {goal: "synthesize across A,B", expected_tool: null}
  difficulty: hard
  category: multi_hop                  # single_hop | multi_hop | unanswerable | adversarial | tool_required
  failure_mode_targeted: "tests whether agent retrieves BOTH papers before answering"
  corpus_hash: "sha256:..."

What makes it hard, stated plainly to speak to it:

    Unanswerables are the most valuable and hardest to write. For an agent the bar is higher: the right behavior is to retrieve, observe the gap, and decline — a trajectory, not just an answer.

    The golden trajectory is a third annotation task (answer + retrieval-ground-truth + competent-path). Not every item needs one — efficiency and tool-selection can be judge-scored — but the items that have it are the highest-signal and the most labor.

    adversarial cases are how you actually test the path metrics rather than asserting they work — questions designed to bait a one-shot on a multi-hop, or a give-up-and-confabulate on a missing second hop.

    The dataset is built after first contact with the real agent. You can't write good trajectory annotations in a vacuum; you write them informed by the failure surface you observed in §9 step 4. The 40% allocation doesn't shrink — it's spent better because it comes after you've watched the agent be dumb.

    Versioning is non-negotiable. Content-hashed, hash logged per run; a drifting golden set makes regression detection meaningless.

5. Observability & tracing (built early, not late)

Per-step spans — thought, tool, args, result, observation, latency, token cost — captured via LangSmith / Phoenix from the very first agent run, so you are never debugging blind and so the failure that justifies the whole eval layer is visible when it happens. What it buys:

    Failures surface with their trace. A flagged trajectory links to its span tree — "here is the exact step where it called the wrong tool." Makes the original "surfaces failures with traces" pitch literally true.

    Per-step cost/latency feeds a real answer to the rare-but-asked cost-modeling interview question (per-conversation cost of an N-turn loop).

    It's a named 2026 skill — tracing a multi-step loop to locate the break — now on the resume with evidence behind it.

6. Regression tracker (production-grade, but small relative to the dataset)

Every /run logs to W&B as an immutable run: all RAGAS metrics, all trajectory metrics, all judge dimensions, per-category and per-step breakdowns, prompt/model/agent versions, retrieval config, golden-set hash, measured variance, aggregate cost/latency.

Baseline diffing. A designated baseline is the reference; each run computes per-metric deltas. If any gated metric drops beyond threshold (default: faithfulness −3%, goal-completion −5%, tool-selection −5%), the run is flagged REGRESSION and /run returns non-zero.

Why per-category and per-step. A 2% aggregate faithfulness drop can hide a 15% collapse on multi_hop. A steady aggregate goal-completion can hide that the agent now takes 5 steps instead of 2 — a real efficiency/cost regression invisible to answer-level metrics. Surfacing both is what makes the "regression I caught" story reproducible from logged runs, not remembered.
7. Service & CI layer (the ~5% — plumbing, on purpose)

FastAPI. /evaluate, /evaluate-trajectory, /run, /runs/{id}. The units other services integrate against; suite runs backgrounded because a 100-item suite with a 3× self-consistency judge over trajectories is minutes.

GitHub Action. On every PR: spin up service, POST /run, block, assert no gated metric breached. A prompt or agent-graph change that quietly tanks faithfulness or goal-completion or tool-selection cannot merge. ~40 lines of YAML — the highest-signal-per-line artifact in the repo, because it's the proof eval is wired into the loop. But it is plumbing: do not spend the project's hours here (see §11).
8. Stack

Python · LangChain · LangGraph · Chroma (or pgvector / Qdrant) · OpenAI API · Anthropic API (judge, cross-family) · RAGAS · DeepEval · LangSmith / Arize Phoenix · FastAPI · W&B · Docker · GitHub Actions · pytest

Each earns its place: RAGAS = answer-level reference-free metrics; DeepEval = judge-assertions + CI test harness; LangGraph = agent runtime (state, checkpointing, tool nodes); LangSmith/Phoenix = tracing; W&B = regression store; two providers = agent/judge family separation (§4.1 Track C). The high-frequency 2026 keyword cluster (LangGraph, agents, tool-calling, vector DB, OpenAI + Anthropic, tracing, FastAPI, Docker, CI/CD) falls out of building the harder thing — it is a consequence of the engineering, not the reason for it.
9. Build order — reversed, failure-first

The v0.2 order built the gate before observing a failure — solving a problem on faith. v0.3 inverts it so the eval exists because a real regression was watched:
#	Step	Proves / produces
1	Minimal LangGraph agent — plan/retrieve/synthesize, one tool, explicit state	A real agent exists to observe
2	Tracing on from run one (LangSmith/Phoenix)	Never debug blind
3	Run against ~10 hand-written questions (incl. 2 multi-hop, 1 unanswerable); watch traces	First contact with reality
4	Break it on purpose / catch it being dumb — one-shots a multi-hop, confabulates a second hop, answers an unanswerable	The seed of the whole project + story #4
5	Golden set v1, aimed at the failure modes actually observed in 3–4 (30 items incl. unanswerables + ~5 trajectory cases)	The moat, built informed not blind
6	RAGAS track — answer-level scores reproducible	Quantitative baseline
7	Trajectory scorer — planning / tool-selection / retrieval-strategy / efficiency / goal-completion	The agentic upgrade — the rare part
8	Cross-family judge + calibration (κ reported)	Differentiator + intellectual honesty
9	W&B logging + baseline diff + per-category/per-step flagging	"Production-grade" earned
10	FastAPI + Docker, then GitHub Action gate	Service + in-the-loop
11	Golden set → 50–100; write up the real regression with its trace	The interview story, finished

Ship is step 9. Steps 5, 7, 8 (dataset, trajectory scorer, calibrated judge) are where signal concentrates — they are the project. Steps 10–11 are real but small. The failure in step 4 is not a phase to rush past; it is the origin the entire harness is built to have.
10. Effort allocation — and why most portfolio projects look the same

Most people invert these numbers — they spend the weekend on the agent, the service, the CI, the trendy surface, and an afternoon on a thin dataset. That inversion is exactly why most portfolio projects are interchangeable. The asymmetry below is the differentiation:
Allocation	Component	Rationale
40%	Golden dataset	The moat. Adversarial multi-hop, retrieval ground truth, trajectory annotations, unanswerables — the days of work almost nobody does.
20%	Trajectory scoring	The rare, agentic part. Wrong tool / unnecessary tool / missed retrieval / inefficient path.
15%	Tracing	Built early; makes failures legible; named 2026 skill.
10%	Regression analysis	Per-category/per-step diffing — produces the story.
10%	Agent	Deliberately mediocre. The subject under test, not the point.
5%	Service + CI	Plumbing. Real, small, do not over-invest.

A mediocre agent that a rigorous harness catches being mediocre is a better demo than a great agent with shallow eval. Depth goes in the dataset and the engine.
11. The governing litmus test

    If I removed FastAPI, Docker, GitHub Actions, and W&B entirely, would the project still be impressive?

For v0.3 the answer is yes — because what remains is a hand-built adversarial golden set, trajectory scoring, and a calibrated cross-family judge, which is the substance. Keep this question pinned to the build. It is not a one-time scope check; it is a build-time discipline. Every hour you're tempted to polish the FastAPI layer or fight a GitHub Action YAML, this question tells you whether you're doing the 5% work or the 40% work. If the answer ever drifts toward "no," the project has lost its center and effort has migrated to the plumbing.

A related self-check, because three problems are bundled here (agent evaluation, observability, regression infrastructure) and any one is a credible project on its own: the sweet spot is where they reinforce each other rather than becoming separate deliverables. Tracing exists to make the evaluated failure legible; regression infra exists to catch what the eval scores; the eval exists to grade the agent the traces expose. If any of the three starts to feel like its own standalone build, it has drifted out of the sweet spot.
12. The interview stories this is engineered to produce

    "Walk me through an evaluation you designed." (The named 2026 screen.) → The three orthogonal tracks, why each, and the calibration/κ work that makes the judge trustworthy.

    "Why isn't RAGAS enough?" → §4.2. Orthogonality + a concrete disagreement case.

    "How do you evaluate an agent, not just an answer?" → §4.1 Track B + §5. Trajectory scoring + trace-level observability. The rare one most candidates can't answer.

    "What regression did you catch?" → §6 + §9 step 4. The planner-prompt change that dropped tool-selection on multi-hop while single-hop held, recovered from W&B with its trace. The story the whole build exists to earn.

    "What does a 10-turn agent loop cost?" → §5. Per-step token/cost logging gives a real number.

    "Faithfulness vs. relevancy — and efficiency vs. completeness — tension." → a real Pareto point from your own runs.

13. Resume line this produces

    Built a continuous evaluation and observability harness for LangGraph-based AI agents — trajectory scoring (planning, tool-selection, retrieval-strategy, efficiency), a calibrated cross-family LLM-as-judge, hand-curated adversarial golden dataset with retrieval and trajectory ground truth, regression tracking, tracing, and a CI deploy gate that blocks merges on metric regression.

Compare: "Built a RAG chatbot over arXiv papers." Same underlying corpus; entirely different signal. The difference is everything in §4, §5, and §10 — the substance the litmus test protects.

Design locked. The next meaningful insight will not come from another architecture document — it will come from the first time the agent does something dumb, the trace explains why, and the evaluator either catches it or fails to. That is where the real project begins. Build steps 1–4 first.
