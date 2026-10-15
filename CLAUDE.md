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

- Code: package `redraw` (26 CPU tests pass). `notebooks/round1.ipynb` trains `redraw`,
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
