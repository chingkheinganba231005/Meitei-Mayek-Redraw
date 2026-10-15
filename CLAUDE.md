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
   enforce this routing.

Related work to cite and differentiate: Revow, Williams & Hinton 1996 (deformable spline digit
models, EM fitting, slow, 10 classes); Hinton, Williams & Revow 1991; Mihai & Hare 2021
(differentiable rasteriser, unsupervised MNIST tracing); Dalca et al. 2019 (conditional deformable
image templates); Deformable ProtoPNet (similarity, not reconstruction); kanji trajectory recovery
(needs online data); InkSight; TRACE; MMI discriminative training of HMMs (speech) as the precedent
for "discriminative class models under best-fit alignment". Re-check novelty before writing.

## Status (October 2026)

- Code: package `redraw` (19 CPU tests pass). `notebooks/round1.ipynb` trains `redraw`,
  `redraw_hdr`, `baseline`, `redraw_norender` (ConvNeXt-T, 40 epochs, validation only).
  `redraw_hdr_e2e` (same network trained end to end, no gradient routing) is the ablation that
  tests the training principle; planned for round 2.
- Checks so far: per-image fitting traces all 55 real class samples with <= 8 strokes; the
  amortised parser learns from rendering losses alone (synthetic); on a toy set of look-alike
  classes (one extra small stroke), the template cost alone picks the right class 66% of the time
  with the fit loss only and 100% with the margin loss.
- Next: read round-1 results (decision: continue if `redraw_hdr` and `redraw` beat `baseline` and
  `redraw` beats `redraw_norender` on validation), then stronger backbone / 60 epochs / 3 seeds,
  test set once, robustness (pen width, blur, low resolution), figures.

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
