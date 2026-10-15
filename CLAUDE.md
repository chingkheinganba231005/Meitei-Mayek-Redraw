# Reading by redrawing: project brief

Owner: Chingkheinganba Rajkumar (GitHub chingkheinganba231005, ORCID 0009-0002-2620-0281),
School of Computing and Data Science, The University of Hong Kong.

Claude Code reads this file at the start of every session. Keep it current: when a decision is
made or a number is final, write it here.

## Goal

A new paper (target: The Visual Computer or a similar visual-computing journal) on handwritten
Meitei Mayek character recognition (TUMMHCD, 55 classes) with a genuinely new mechanism,
architecture and learning methodology, and a test accuracy above **98.12%**.

98.12% is the owner's earlier result (six fine-tuned ImageNet CNNs, plain average; single networks:
ConvNeXt-T 97.83%, EfficientNetV2-S 97.95%, ResNet-50-D 97.64%). That paper was desk-rejected by
The Visual Computer for lack of methodological novelty and has been transferred to another journal,
where it is under consideration. This project must stay clearly distinct from it: new code, new
experiments, no reused text or figures, and the earlier paper cited (or declared as related work
under consideration when submitting). Best published result by others: 97.08% (Hijam & Saharia).

## The method

1. **Free stroke parsing (architecture).** A pretrained CNN (timm, unchanged) feeds an image head.
   A stroke decoder (K=12 queries over the stride-8/16 feature maps) predicts cubic Bezier strokes
   with width and existence. A differentiable renderer draws them and the drawing is compared with
   the input: the only supervision the strokes get (no stroke labels, no online data). A stroke head
   (set transformer over stroke tokens: geometry, width relative to the glyph box, CNN features
   sampled along the curve) classifies the strokes. Output = image logits + stroke logits.
2. **Hypothesis-driven redrawing (mechanism).** Every class has M=2 learned templates of S=8
   strokes. For the top-3 candidates, a fitter bends each candidate's templates onto the image
   (affine + bounded control-point offsets + width/existence changes); the render-and-compare cost
   is the evidence for that class and re-ranks the candidates.
3. **Honest analysis-by-synthesis training (learning methodology).** Gradient routing:
   fit loss trains the fitter for every hypothesis (best effort, right or wrong) and only the true
   class's templates; a margin loss with the fitter frozen trains only the templates (the right
   class must redraw the image better than each wrong candidate by a margin); a plain cross-entropy
   on the re-ranked logits (fused logits held fixed) trains templates and the two scalars. The
   fitter is never rewarded for fitting a wrong class badly. Tests in `tests/test_hypotheses.py`
   enforce this routing. `honest=False` (config `redraw_hdr_e2e`) is the ablation.
4. **StrokeWarp (learning methodology).** Self-structured augmentation: half of each batch is
   deformed along the parse of the network's own weight-averaged copy. Inner control points move
   (curvature changes, end points fixed so joined strokes stay joined), free tails grow or shrink
   along their own direction, pen width never changes; each output pixel samples the real ink back
   along a softmax-weighted field over the bent stroke samples. Starts after `warmup` epochs.
   (`redraw/strokewarp.py`, config `redraw_hdr_sw`.) A first version with rigid per-stroke
   rotation/scale/shift tore junctions and thickened strokes, hence bends and tails only.

Framing for the paper (owner's concern after the desk rejection): lead with the principle (honest
evidence: a recogniser whose redrawing evidence cannot be gamed) and the two learning methods, not
with the stroke branch, which on its own reads as an extra module on a CNN. Ancestors to cite:
elastic/deformable templates (Hinton, Williams, Revow 1991-96), Bayesian Program Learning (Lake et
al. 2015), latent-SVM deformable part models (Felzenszwalb et al. 2010), render-and-compare
hypothesis scoring in 6D pose (e.g. MegaPose), stroke warping on online data (Gurmukhi, ICFHR 2018).
No prior work found on the honest gradient routing or on parse-guided stroke-wise warping of
offline images (searched October 2026).

Related work to cite and differentiate: Revow, Williams & Hinton 1996 (deformable spline digit
models, EM fitting, slow, 10 classes); Hinton, Williams & Revow 1991; Mihai & Hare 2021
(differentiable rasteriser, unsupervised MNIST tracing); Dalca et al. 2019 (conditional deformable
image templates); Deformable ProtoPNet (similarity, not reconstruction); kanji trajectory recovery
(needs online data); InkSight; TRACE; MMI discriminative training of HMMs (speech) as the precedent
for "discriminative class models under best-fit alignment". Re-check novelty before writing.

## Status (October 2026)

- Code: package `redraw` (52 CPU tests pass, version 0.5.0). `notebooks/round1.ipynb` trains `redraw`,
  `redraw_hdr`, `baseline`, `redraw_norender` (ConvNeXt-T, 40 epochs, validation only).
  `notebooks/round2.ipynb` reuses those and adds `redraw_hdr_sw` (StrokeWarp), `redraw_hdr_e2e`
  (honest-training ablation) and `redraw_hdr_nohead` (no stroke head), with `verify_eval`
  (top-3 recall, cost-only accuracy, rescued/broken counts) and figures.
- Round-1 configs must not change while the owner may be running round 1 from `main`:
  checked by training all four on fake data with the first commit and the current code
  (identical histories).
- Checks so far: per-image fitting traces all 55 real class samples with <= 8 strokes; the
  amortised parser learns from rendering losses alone (synthetic); on a toy set of look-alike
  classes (one extra small stroke), the template cost alone picks the right class 66% of the time
  with the fit loss only and 100% with the margin loss.
- Next: read round-1 results (decision: continue if `redraw_hdr` and `redraw` beat `baseline` and
  `redraw` beats `redraw_norender` on validation), then stronger backbone / 60 epochs / 3 seeds,
  test set once, robustness (pen width, blur, low resolution), figures.

## Results so far (validation, 5-view TTA, ConvNeXt-T, 40 epochs, one seed)

| network | val acc | errors | note |
|---|---|---|---|
| baseline | 97.90% | 227 | plain fine-tuning |
| redraw | 97.84% | 234 | parse F 0.93 (recall 0.9998, precision 0.88), ~10 strokes |
| redraw_norender | 97.98% | 219 | best, but the control without rendering |
| redraw_hdr | 97.78% | 240 | check: top-3 recall 99.8%, cost alone 93.3%, rescued 1, broken 1 |
| redraw_hdr_sw | 97.80% | 238 | StrokeWarp: no gain (McNemar 57 vs 55) |
| redraw_hdr_e2e | 97.87% | 231 | end to end not worse than honest (52 vs 43, p = 0.41); cost alone 91.6% |
| redraw_hdr_nohead | 97.79% | 239 | rescued 9, broken 10 |

All differences are within noise (McNemar p >= 0.17). Conclusion: unsupervised stroke parsing
works (a real capability), but neither it, the template check, StrokeWarp nor honest routing adds
accuracy on this in-distribution benchmark; single networks sit at ~97.9% and 98.12% came from a
six-network ensemble. Bug found afterwards: rounds 1-2 computed the classification loss in
bfloat16 (outside autocast); fixed in 0.3.1 (float32 loss). Comparisons within rounds 1-2 are fair,
absolute values may be slightly low. Next: `notebooks/diagnostics.ipynb` (float32 baseline,
diversity of the eight networks, nearest-neighbour memory with two-fold held-out tuning) to decide
the next mechanism.

## Diagnostics (validation) and what they mean

- Float32 baseline 97.91%; the average of the eight trained networks 98.01%; some network is right
  on 98.92%; 117 images are wrong for all eight (74 of them in the four hard pairs).
- Errors are concentrated in four pairs; I (025) vs I lonsum (044) alone accounts for 76 validation
  errors and 78 of the 241 test errors of the owner's 98.12% ensemble. The size features carry no
  information on that pair (the earlier project's size probe equals the majority rate).
- Nearest neighbours in feature space: median top-1 cosine 0.996 (near-duplicates, same writers);
  on classifier errors the top-1 neighbour has the true label only 7.5% of the time and median
  similarity 0.91. A kNN memory adds nothing (held-out gain about 0). The remaining errors are label
  conflicts the image cannot settle, not missed shape cues.
- Cover-Hart with the 1-NN error (2.11%): Bayes error in that feature space >= 1.06%, i.e. no
  classifier on those features exceeds about 98.9% on validation.
- Every coupled stroke component lowered the network's own image head (redraw 97.73, hdr 97.61 vs
  97.90): the backbone pays for learning to redraw.
- The 98.12% models were already trained on train+val (`run_dir/<name>/full`), so retraining on
  train+val gives no edge over them; the margin must come from stronger single networks.

## Round 3 (validation only, `notebooks/round3.ipynb`, code 0.4.0)

- `route="honest"` (`redraw_hdr_hr`): the stroke decoder and the hypothesis fitter read the backbone
  through a stop-gradient; the stroke head reads features along the strokes but cannot move them.
  The backbone learns from classification alone. Together with the honest training of the check this
  is the paper's learning methodology: every loss reaches only the parts it is meant to explain.
  Tests in `tests/test_routing.py`.
- Backbone pilot at 128 px (`baseline_in12k`, `baseline_v2`), then 224 px with the best one
  (`baseline224`, `redraw_hdr_hr224`; `res` resizes the 128 px cache on the GPU).
- Test-time redrawing (`train.predict_ttr`): average over the input and n StrokeWarp re-writings
  made from the network's own parse (fixed draws). Only possible for a network that parses strokes.
- Diversity (pairwise averages), scarce data (2/5/10%, 2000 steps, baseline vs coupled vs honest),
  damaged scans (7 corruptions x 2 levels).
### Round-3 results (validation, 5-view TTA, one seed each)

| network | final | image head | stroke head | errors | note |
|---|---|---|---|---|---|
| baseline_fp32 | 97.91% | 97.91% | - | 226 | control |
| redraw_hdr (coupled) | 97.78% | 97.61% | 97.40% | 240 | |
| redraw_hdr_hr (honest) | 97.86% | 97.73% | 97.46% | 232 | vs control 54/60, p = 0.64; check +4/-1, cost alone 93.7% |
| baseline_in12k | 97.65% | | | 254 | worse |
| baseline_v2 | 97.83% | | | 235 | not better |
| baseline224 | 97.66% | | | 253 | worse than 128 px (57/84, p = 0.03): peaks at epoch 20-25, then overfits |
| redraw_hdr_hr224 | 97.87% | 97.63% | 97.28% | 231 | holds at 224 px (vs baseline224 71/49, p = 0.055) |

- Honest routing recovers about half of the image-head cost (97.61 -> 97.73, control 97.91) and
  matches the control on the final output; no significant difference either way.
- Test-time redrawing: no gain at 128 px; at 224 px one view 97.80 -> 97.91 (18/6, p = 0.02).
- Average of all 13 networks 98.09%; best pairs 98.04-98.06. Mixing families does not add more
  diversity than another seed (mean pair gain: coupled-plain +0.16, plain-plain +0.14,
  honest-plain +0.10).
- Scarce data (one seed): honest beats control at every share: 2% 94.19 vs 94.04, 5% 95.54 vs
  95.35, 10% 96.12 vs 95.95; coupled in between; the check rescues more at 2% (+12/-8 honest, +14/-6 coupled).
- Damaged scans: learning to redraw with coupled routing gives the backbone a shape bias.
  Heavy blur 93.28 vs 90.20, 12 px 90.03 vs 86.23, rotation 20 degrees +0.4; but heavy noise 93.52
  vs 95.56, uneven light 84.49 vs 88.97, salt and pepper 90.04 vs 92.99. Honest routing sits in
  between. Mean over the 14: control 94.39, coupled 94.28, honest 94.25.

### Assessment after round 3

The random split is saturated: every recogniser tried (8 designs, 3 backbones, 2 resolutions)
lands at 97.65-97.98% on validation, the differences are inside noise except the two worse
pretrainings and 224 px, and averages of many networks stop at about 98.1%. The earlier 98.12%
(six networks) is at that ceiling. Expected test results of the final run (from validation, and
val -> test was +0.04 for the earlier ensemble): single networks about 97.9%, average of the
method's three seeds about 98.0%, all seven about 98.05-98.1%; above 98.12% is possible but not
likely, and no difference of that size can be significant (95% interval +-0.24 points). The paper
must therefore rest on the mechanism and methodology, the analysis of the ceiling, and where
structure helps (unfamiliar handwriting, scarce data, blur and low resolution), with the
standard benchmark reported honestly (above the best published 97.08% by about 0.8 points).

### Final run (`notebooks/final.ipynb`)

Fixed before looking at test: ConvNeXt-T in22k, 128 px, 40 epochs; `redraw_hdr_hr` (method) x 3
seeds, `baseline_fp32` (control) x 3 seeds, `redraw_hdr` (coupled) x 1, all on train+val (tag
`full_s<seed>`). Test once: singles (mean +- s.d.), seed averages, McNemar, bootstrap intervals,
hard pairs, parse, check, test-time redrawing; accuracy by familiarity (most similar training
image on 32 x 32 thumbnails, `analysis.familiarity`; bins: least familiar 5%, 5-10%, 10-25%,
rest), also on validation for the round-3 networks; damaged scans on test; scarce data with seeds
1-2 added and all three evaluated on test.

## Final results (test set, 12,794 images, 5-view TTA; `MyDrive/redraw/final/results.json`)

| network | singles (3 seeds) | mean +- s.d. | seed average | errors |
|---|---|---|---|---|
| control (`baseline_fp32`) | 97.76 / 97.83 / 97.90 | 97.83 +- 0.07 | 97.88 (CI 97.63-98.13) | 271 |
| ours (`redraw_hdr_hr`) | 97.90 / 97.94 / 97.83 | 97.89 +- 0.06 | 97.94 (CI 97.69-98.18) | 264 |
| coupled (`redraw_hdr`, 1 seed) | 97.84 | | | 276 |

- McNemar ours vs control per seed: 82/64 (p 0.16), 82/68 (0.29), 53/62 (0.46). Over 12 paired runs
  (3 full + 9 scarce-data) ours ahead in 10, mean +0.08 points, Wilcoxon p = 0.007 (conditional on
  the test set). 98.12% (earlier six-network ensemble) is NOT beaten; the paper does not claim it.
- Ours: image head 97.76; stroke reader 97.29-97.55; parse F 0.930-0.931 (P 0.878, R 0.9999), ~9.7
  strokes; check: top-3 recall 99.84%, energy alone 93.6%, rescued 11 / broken 14 over 3 seeds;
  test-time redrawing no gain. Cost: 35.6 M vs 27.9 M parameters, 3.70 vs 2.90 GFLOPs, 3.1x training.
- Scarce data (test, 3 seeds): 2% 94.06/94.14/94.14 (control/coupled/ours), 5% 95.34/95.35/95.45,
  10% 95.99/96.00/96.08.
- Damaged scans (stronger level, vs control): coupled +3.05 blur, +2.77 at 12 px, +1.02 at 16 px,
  -1.70 noise, -2.60 salt and pepper, -5.45 uneven light; ours same signs, smaller. Mean over 14
  conditions: 94.11 / 94.01 / 93.97 (control / ours / coupled).
- Errors: four pairs = 54-59% of every network's errors (1,802 test images); 130 test images wrong
  for all 7; pixel nearest neighbour 86.6% (52.7% on the least familiar 5%), networks flat 97.3-98.0%.

## Paper (draft v1, not in the repository)

"Reading by redrawing: stroke-level explanations for handwritten Meitei Mayek recognition at no cost
in accuracy", Springer sn-jnl, 23 pages, built from the results files only (figures by
`make_figs.py`); Algorithm 1 gives one training step with the stop-gradients. Claims: a mechanism
(unsupervised Bezier parse + template check), a training principle (honest gradient routing; cites
Cloud et al. 2024 "gradient routing" and concept-bottleneck leakage as related), a shape-bias dial, and
an error analysis; no claim to beat 98.12%. The earlier paper is cited as under review; the cover
letter declares it and its earlier desk rejection at The Visual Computer. Paired tests are stated as
conditional on the shared test set. v2 will add the round-5 results (erasure test, Devanagari, dial,
selective prediction, template and vector figures).

## Round 5 (`notebooks/extras.ipynb`, code 0.5.0), for reviewers

A: TUMMHCD final networks, inference only: learned templates of all 55 classes, vector redrawing +
SVG export, selective prediction (softmax vs energy margin vs rank average; AURC, risk at 90/95/98%).
A4 (`redraw/erase.py`): erase one stroke of each test image (pixels nearest that stroke's edge, within
0.04) chosen by the stroke reader (most / least important: leave-one-out drop of the reader's
log-probability of the final class), at random, the largest, or by gradient x input of the reading
network; ours and the control read every erased image; seeds paired.
B: routing dial `draw_grad` 0.25 and 1.0 (`redraw_hdr_hr_g25`, `_g100`), full data, test + damaged
scans. C: Devanagari (DHCD, UCI, 46 classes), recipe unchanged: control x3, ours x3, coupled x1,
test once, damaged scans, 2/5% x3 seeds, parses and templates. Results go to `MyDrive/redraw/extras/`.

## Data

TUMMHCD from its maintainers (http://agnigarh.tezu.ernet.in/~sarat/resources.html). In the owner's
Google Drive: `MyDrive/tummhcd98/TUMMHCD-TEST-TRAIN.zip` (found automatically). This project's
outputs go to `MyDrive/redraw/`. Split: 15% of each training class for validation, seed 42
(61,504 / 10,826 / 12,794). Choose everything on validation; touch the test set once.

## Working rules

- Experiments run in Google Colab on one A100 (80 GB). Every reported number must come from a
  results file written by the code.
- Commits: author `chingkheinganba231005 <chingkheinganbaofficial@gmail.com>`.
- Never ask for tokens or passwords in chat; they go in Colab secrets.
- The owner's older TACL-Net project is unpublished and private: do not mention it anywhere.
  (This method shares nothing with it: no skeleton inputs, no skeleton graph, no pair-contrastive loss.)
- Keep `docs/ai_use_log.md` current (what Claude Code did, dated) so the journal's AI statement is
  accurate.
- Writing: plain, precise, British spelling, no filler.
