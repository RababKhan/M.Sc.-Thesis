# Terminology audit (Abstract, Chapters 1–8, Appendices)

Method: each term family was counted per file with case-sensitive and case-insensitive searches. Text inside code
formatting (file names, configuration keys) was excluded wherever it would distort the count. Every variant was then
read in context. No chapter text was changed. Results are graded:
- **CONSISTENT**: one form, or variants follow a stated rule;
- **ADVISORY**: harmless variation that could be harmonised at template stage;
- **FIXED**: corrected in material produced in this stage (Abstract, appendices). Approved chapters were not touched.

## 1. Spelling convention

| Family | British forms (prose) | American forms (prose) | Verdict |
|---|---|---|---|
| optimise / optimisation / optimiser | 52 | 0 | CONSISTENT |
| centred | 49 | 0 | CONSISTENT |
| normalise / normalised | 33 | 0 | CONSISTENT |
| behaviour | 31 | 0 | CONSISTENT |

The only American form, `optimizer` in Appendix A.3, is a literal configuration key copied from
`phase6_config.json`. It is now shown in code formatting with the other keys (**FIXED**: presentation only).

**ADVISORY: title spelling.** The registered thesis title uses American spelling ("…Sensitivity-Aware
Optimization"), while the body is consistently British ("Proximal Policy Optimisation" in the Abstract and
keywords). This is common and acceptable when a title is fixed by registration, but some institutions require one
convention throughout. Two options:
- (a) keep the registered title verbatim and the body British (current state);
- (b) follow whatever the official template or regulations require.

The thesis title was not changed. Reference entries keep each source's original spelling, e.g. "Proximal Policy
Optimization Algorithms".

## 2. Method and baseline names

| Term | Usage found | Verdict |
|---|---|---|
| Uniform / uniform pruning | Capitalised as a method label (tables, "Uniform-Structured", "PPO–Uniform"); lower case in running prose ("uniform pruning", "uniform allocation") | CONSISTENT (label vs prose rule) |
| Random / random allocation | Same rule as Uniform | CONSISTENT |
| global magnitude / Global | "global magnitude" in chapter prose (31 uses); "Global" as a short table or comparison label ("PPO–Global", Chs 2, 4, 5); hyphenated "global-magnitude" only as a compound adjective (Ch. 1 ×2, Ch. 3 ×1) | CONSISTENT. **ADVISORY:** Ch. 8 Table §8.3 row C1 mixes styles in one list ("Uniform, Random, global magnitude, ERK and LAMP"). It is correct, but a template editor may prefer one style |
| LAMP, ERK, HALP | Acronym form everywhere; expanded once in `list_of_abbreviations.md` | CONSISTENT |
| PPO | Expanded at first use in the Abstract and Ch. 1; acronym thereafter | CONSISTENT |

## 3. Recovery and fine-tuning

"Equal recovery" is defined as identical **fixed-mask fine-tuning** (Ch. 3 §3.15, Ch. 4 §4.13 "Recovery and
Fine-Tuning Protocol"). The two terms are used as synonyms by design: "recovery" for the step in the pipeline,
"fine-tuning" for the training procedure. Ch. 1 RQ5 and Ch. 5 §5.10 ("Equal Fine-Tuning") use the fine-tuning form,
and Chapter 8 answers RQ5 with "fixed-mask recovery". Counts: recovery 184, fine-tun(e/ing) 97.

Verdict: **CONSISTENT** (defined synonym pair). Each chapter introduces the pair before alternating. There is one
non-hyphenated form, `finetune`, a function name in code formatting in Ch. 3 §3.15.

## 4. Name of the Phase-7B agent

| Form | Where |
|---|---|
| "structured, energy-aware PPO" | Ch. 3 (Table 3.1, §§3.5, 3.16, 3.16.3, Algorithm 3.2), Ch. 4 Table 4.7, Ch. 6 §6.10 heading and Table 6.5, Ch. 7 §7.10 |
| "deployment-oriented structured PPO / agent" | Abstract, Ch. 1 (×3), Ch. 7 (×2), Ch. 8 (×2) |

Both name the same agent. "Energy-aware" describes its *reward*, which uses the surrogate-predicted energy saving
(Eq. 3.19). It does not imply a measured energy result, and every chapter states that 7B energy is unresolved.
"Deployment-oriented" is the framing used in the summary chapters.

Verdict: **ADVISORY**. No claim conflict. If the examiners prefer one name, use "deployment-oriented structured PPO
(energy-aware reward)" at first mention in Ch. 6 §6.10 at template stage. That would be a wording change requiring
approval.

## 5. Sustainability and environmental wording

- "Sustainability" appears only as an assessment frame (Ch. 6 title, §6.14 "Three-Level Sustainability Assessment",
  and Ch. 7–8 references to it). It never appears as a claim.
- "green" appears in prose only as:
  - the cited external agenda "Green AI" (Ch. 2);
  - the internal criterion name GREEN-SUCCESS, labelled as an internal file name (Ch. 6, Appendix I);
  - literal folder paths in code formatting.
- "environmentally friendly", "eco-friendly", "carbon reduction" and "lower carbon" each occur 0 times in claims (see
  `global_claim_audit.md`).

Verdict: **CONSISTENT** with the standing rule that no green, environmental or carbon claims are made.

## 6. Labels that reuse a letter

| Label | Meanings | Verdict |
|---|---|---|
| **C1–C6** | Contributions 1–6 (Ch. 1 §1.7, Table 1.1; Ch. 8 §8.3) **and** Phase-5 conditions C0–C2 (Ch. 4 §4.9, Ch. 5 §5.8, Ch. 7, e.g. Ch. 7 l. 77 "C1's test accuracy") | **ADVISORY.** The chapters never use both meanings, and each chapter defines its own. Recorded in `list_of_abbreviations.md`. Template-stage option: write the Phase-5 conditions as "Condition C1" at first use in Ch. 7 |
| **S1/S2, B-, D-, E-families** | Phase-4 variant and family labels | CONSISTENT (defined in Ch. 4 §§4.8–4.10, Table 4.6) |
| **ρ** | sparsity; Spearman ρ in Ch. 6 | ADVISORY, see `final_notation_inventory.md` |

## 7. Splits, phases and units

| Term | Verdict |
|---|---|
| TRAIN-RL, VAL-RL, VAL-SELECT, TRAIN-FT; configuration (I)/(II) | CONSISTENT (`split_governance_audit.md`) |
| "Phase 7A" (noun) / "Phase-7A" (compound adjective, e.g. "Phase-7A energy") | CONSISTENT (hyphenation rule) |
| pp / percentage points | "percentage points" spelled out in the Abstract, Ch. 1 and Ch. 8 summaries; "pp" elsewhere, defined in the abbreviations list | CONSISTENT |
| top-10 (adjective) / top 10 (noun phrase) | CONSISTENT |
| pruning unit / pruning ratio | CONSISTENT in Chs 1–8. The Abstract said "each layer unit", a form used nowhere else. **FIXED:** now "each layer or layer group" (accurate for ResNet-8's residual-stage units; Abstract 349 words) |
| "sparsity ratio" (Ch. 2 ×2) | Describes the action space of cited methods (AMC, QLP), not this thesis | CONSISTENT |
| retained accuracy (q) | CONSISTENT |

## 8. Model and dataset names

CIFAR-10, CIFAR-100, SimpleCNN, LeNet-5 and ResNet-8 are written identically in all prose. The forms CIFAR10, ResNet8
and LeNet5 occur only inside code-formatted file names and identifiers (Appendices, one each in Chs 4 and 6).
Verdict: **CONSISTENT**.

## 9. Framework components and deployment quantities (Part N list)

Counts are prose occurrences (code formatting excluded), in the Abstract and Chapters 1–8.

| Term in Part N | Forms found | Verdict |
|---|---|---|
| sensitivity prior | "sensitivity prior" (16), "action prior" (Ch. 3 ×2, in the §3.9 heading sense) | CONSISTENT |
| centred sensitivity prior | "centred sensitivity prior" at first use (Abstract, Ch. 2, Ch. 3 ×3), then the short form "centred prior" (21) | CONSISTENT (long form first, short form after) |
| archive selection | "archive selection" (23), "archive-based selection" (7), "archive-based policy selection" (Ch. 3 heading §3.12, Ch. 8 RQ3 = the RQ wording) | CONSISTENT (the long forms follow the RQ3 and §3.12 headings) |
| policy retention | "policy retention" (7), "retention problem" (7) | CONSISTENT |
| constrained reward | "accuracy-constrained reward" (16) at first use, then "constrained reward"; "constraint-first reward" (5) is used only where its *form* is described (Ch. 1 C4, Ch. 3 §§3.13 and 3.16, Ch. 6, Ch. 8) | CONSISTENT |
| equal recovery | "equal recovery" (26), "equal post-pruning recovery/fine-tuning" (6, the RQ5 wording), "equal fine-tuning" (7, the §5.10 heading and its references) | CONSISTENT (synonym pair, §3 above) |
| masked pruning | "masked pruning" (6); also "unstructured pruning" (Ch. 2 literature ×2; Ch. 3 Table 3.3 caption) and "mask-based" (Ch. 4 ×1). Ch. 3 Table 3.3 defines the core framework as unstructured (masked) | CONSISTENT |
| structured pruning | "structured pruning" (31) | CONSISTENT |
| physical compaction | The phrase does not occur. The chapters say "physically compact(ed)" (Ch. 3 ×3, Ch. 4 ×3, Ch. 6 ×1), "structured compaction" (Ch. 4 H1) and "channel removal" (4). All describe the same operation, defined in Ch. 3 §3.16 | CONSISTENT. No new term is needed |
| deployment-oriented structured PPO | see §4 above | ADVISORY |
| operational energy | "operational (inference) energy" (11), used for the *quantity* the thesis is about; Ch. 1 §1.8 and Ch. 8 §8.5 bound it as "not an environmental assessment" | CONSISTENT |
| package energy | "CPU package energy" (20), used for the *measured* RAPL quantity; Ê is "predicted/surrogate energy" | CONSISTENT. The measured and predicted quantities are never mixed |
| parameter reduction | "parameter reduction" (21) as the latency-proxy cost metric (Paper 3) | CONSISTENT |
| memory reduction | "parameter memory" / "parameter-memory reduction" (23), "memory reduction" (4, all in Ch. 4 l. 580 and Ch. 6 §6.11, where "parameter memory" is defined in the preceding sentence) | CONSISTENT. Memory always means parameter memory; activation memory is never claimed |
| MAC reduction | "MAC reduction" (24); FLOPs defined once as 2 × MACs (Ch. 4 Table 4.8) and otherwise used only for literature (Ch. 2) | CONSISTENT |
| latency | "latency" (206), always measured batch-1 CPU latency in this thesis's results; T(**r**) in equations | CONSISTENT |
| speed-up | "speed-up" (30), always hyphenated, defined in Ch. 4 Table 4.8 as T_ref / T (reported with ×), alongside latency gain g_T = 1 − T/T_identity | CONSISTENT ("speedup" and "speed up" occur 0 times) |

## Result

There are no terminology errors that affect meaning. There are four advisories, all presentation-level and none
applied to approved chapters:
1. title spelling vs British body;
2. two names for the Phase-7B agent;
3. the C1–C6 label reuse;
4. ρ for Spearman correlation in Ch. 6.

Two fixes were applied, in material produced in this stage only:
- the Abstract's "layer unit";
- code formatting of the configuration keys and file paths in the appendices.
