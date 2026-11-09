# Interview points

**"Walk me through the project."** Synthetic prompts for a store assistant, candidate answers from two
small open models, labels from a code rubric, Bradley-Terry reward models, then SFT, DPO and ORPO on a
135M model with my own LoRA on CPU. Everything is measured once and replayed in CI.

**"Did DPO work?"** By the rubric, yes: it beats base on 82.0%<!-- m:rubric_winrate/dpo.vs.base/win_rate/rate -->
of decided test prompts. But explicit refusals of unsafe requests went from 100% to
0.0%<!-- m:variants/test/dpo/refuses_unsafe/rate -->; it answered them with a copied context line. A
single win rate would have shipped that. I added a release rule on dev (safety first), which picked ORPO.

**"Why would DPO drop refusals?"** Only 54 of 255 train prompts are unsafe, the rejected answers there
were often apologetic too, and DPO pushes away from the rejected answer's wording. Without a term that
keeps the chosen answer likely (ORPO has one; SFT before DPO only partly helps) the model drifted to a
short safe-looking line. I did not retune after seeing it; that is listed as next work.

**"How do you know the reward model isn't just measuring length?"** I report two length rules as
baselines. "Shorter wins" reaches 72.1%<!-- m:reward/shorter_wins/accuracy/rate -->, because chosen
answers are shorter in this data. The reward models beat it (83.2%<!-- m:reward/bow/accuracy/rate --> and
80.2%<!-- m:reward/embed/accuracy/rate -->), and the probes show where they still fail: ID stuffing and
wrong numbers fool them about a quarter of the time.

**"Can you use an LLM judge instead of the rubric?"** Not this one. Run in both orders, it changed its
mind on 82.7%<!-- m:judge_pairs/position_flip/rate --> of pairs and picked answer A
90.7%<!-- m:judge_pairs/picks_first/rate --> of the time. Running both orders and reporting kappa is how
you find that out cheaply.

**"Why hand-write LoRA?"** About 60 lines, tested: B starts at zero so the model is unchanged at step
0, switching adapters off gives DPO's reference model without a second copy, and merging keeps outputs.

**"What would you do with a GPU and a week?"** More unsafe pairs, a KL or SFT term in DPO, several
seeds, a larger model, a stronger judge with the same bias checks, and human labels on a sample to
measure the rubric itself.
