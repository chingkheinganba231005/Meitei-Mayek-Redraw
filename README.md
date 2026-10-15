# Reading by redrawing: handwritten Meitei Mayek recognition

Work in progress. A recogniser for the 55 classes of handwritten Meitei Mayek
in TUMMHCD that, besides classifying the image, redraws the character as a
small set of pen strokes and classifies those strokes too.

- A pretrained CNN reads the image (the **image head**).
- A **stroke decoder** reads the CNN's feature maps and predicts up to K
  strokes, each a cubic Bezier curve with a width and an existence
  probability.
- A differentiable **renderer** draws the strokes, and the drawing is
  compared with the input. That comparison is the only thing that teaches the
  decoder what a stroke is: there are no stroke labels and no online
  (pen-trajectory) data for Meitei Mayek.
- A **stroke head** turns each stroke into a token (its shape, its size
  relative to the glyph, and the CNN features along it) and classifies the
  set of strokes with a small transformer.
- The output adds the logits of the two heads.
- A **hypothesis check** then redraws the image as each of the top 3 candidate
  classes, starting from that class's learned stroke templates, and re-ranks
  the candidates by how well each redrawing explains the image. The fitter that
  bends the templates is only ever trained to fit as well as it can; only the
  templates learn to tell classes apart, so the redrawing costs stay honest evidence.
- **StrokeWarp** deforms training images stroke by stroke along the network's
  own parse: strokes bend, free tails grow or shrink, joints and pen width stay.

## Layout

| Path | What |
|---|---|
| `redraw/render.py` | stroke rendering, reconstruction losses, parse score |
| `redraw/model.py` | the network and the experiment configurations |
| `redraw/train.py` | training loop (EMA, resumable checkpoints), prediction, parse evaluation |
| `redraw/split.py`, `preprocess.py`, `data.py`, `augment.py` | data split, preprocessing, GPU augmentation |
| `redraw/strokewarp.py` | StrokeWarp |
| `notebooks/round1.ipynb` | first experiment: the method (with and without the hypothesis check) vs. plain fine-tuning vs. strokes never compared with the ink |
| `notebooks/round2.ipynb` | second experiment: StrokeWarp, honest vs. end-to-end training, the check without the stroke head |
| `tests/` | CPU tests (`python -m pytest`) |

## Data

TUMMHCD from its maintainers (http://agnigarh.tezu.ernet.in/~sarat/resources.html).
Validation is 15% of each training class, seed 42 (61,504 / 10,826 / 12,794 images).

## Licence

MIT.
