import numpy as np
import pytest
import torch

from redraw.render import render
from redraw.robustness import FRAMED, TEXTURE_SHAPE, corrupt_framed, crop_failures, texture_shape


def ink():
    ctrl = torch.tensor([[[[0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.3]],
                          [[0.5, 0.2], [0.5, 0.45], [0.5, 0.65], [0.5, 0.85]]]]).repeat(3, 1, 1, 1)
    img, _, _ = render(ctrl, torch.full((3, 2), 0.08), torch.ones(3, 2), G=128, T=64, exact=True)
    return img.unsqueeze(1)


@pytest.mark.parametrize("kind", list(FRAMED))
def test_framed_corruptions_are_repeatable_and_in_range(kind):
    x = ink()
    for level in FRAMED[kind]:
        a, b = corrupt_framed(x, kind, level, seed=3), corrupt_framed(x, kind, level, seed=3)
        assert a.shape == x.shape and torch.equal(a, b) and a.min() >= 0 and a.max() <= 1
        assert not torch.equal(a, x)


def test_texture_changes_keep_the_shape_and_shape_changes_keep_the_ink():
    x = ink()
    mask = x > 0.5
    outline = texture_shape(x, "outline", 2)
    assert ((outline > 0.5) & ~mask).sum() == 0 and (outline > 0.5).sum() < 0.8 * mask.sum()
    stripes = texture_shape(x, "stripes", 8, seed=1)
    assert stripes.sum() < 0.8 * x.sum() and ((stripes > 0.05) & ~(x > 0.05)).sum() == 0
    warped = texture_shape(x, "elastic", 0.03, seed=1)
    assert abs(warped.sum() - x.sum()) < 0.2 * x.sum() and not torch.allclose(warped, x)
    assert set(TEXTURE_SHAPE) == {"outline", "stripes", "elastic"}


def test_crop_failures_counts_empty_ink_masks():
    meta = np.array([[24, 24, 10, 10, 0.2], [24, 24, 24, 24, 0.0]])
    assert crop_failures(meta) == 1
