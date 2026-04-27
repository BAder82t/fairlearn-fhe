"""Encrypted-aware ``make_derived_metric``.

Mirrors :func:`fairlearn.metrics.make_derived_metric` but routes
through our :class:`MetricFrame` when ``y_pred`` is encrypted, so the
returned callable transparently supports both modes.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable, List, Union

import fairlearn.metrics as _fl

from ..encrypted import EncryptedVector
from ._metric_frame import MetricFrame

_TRANSFORMS = ("difference", "group_min", "group_max", "ratio")
_RESERVED = ("method",)


class _EncryptedDerivedMetric:
    def __init__(
        self,
        *,
        metric: Callable[..., float],
        transform: str,
        sample_param_names: List[str] | None,
    ):
        if not callable(metric):
            raise ValueError("metric must be callable")
        if transform not in _TRANSFORMS:
            raise ValueError(f"transform must be one of {_TRANSFORMS}")
        sig = inspect.signature(metric)
        for r in _RESERVED:
            if r in sig.parameters:
                raise ValueError(
                    f"callables which accept a {r!r} argument may not be passed; "
                    "use functools.partial()"
                )
        self._metric = metric
        self._transform = transform
        self._sample_params = sample_param_names or []

        # Plaintext fallback — preserves Fairlearn's exact behaviour when
        # y_pred is plaintext.
        self._plain = _fl.make_derived_metric(
            metric=metric, transform=transform,
            sample_param_names=self._sample_params,
        )

    def __call__(self, y_true, y_pred, *, sensitive_features, **other_params):
        if not isinstance(y_pred, EncryptedVector):
            return self._plain(
                y_true, y_pred,
                sensitive_features=sensitive_features,
                **other_params,
            )

        sample_params = {}
        bound = {}
        transform_kwargs = {}
        for k, v in other_params.items():
            if k in self._sample_params:
                sample_params[k] = v
            elif k in _RESERVED:
                transform_kwargs[k] = v
            else:
                bound[k] = v

        dispatch = functools.partial(self._metric, **bound)
        dispatch.__name__ = self._metric.__name__

        frame = MetricFrame(
            metrics=dispatch,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            sample_params=sample_params,
            allow_decrypt=True,  # arbitrary user metric → fall through
        )

        if self._transform == "difference":
            return frame.difference(**transform_kwargs)
        if self._transform == "ratio":
            return frame.ratio(**transform_kwargs)
        if self._transform == "group_min":
            return frame.group_min()
        if self._transform == "group_max":
            return frame.group_max()
        raise ValueError(f"unknown transform {self._transform!r}")


def make_derived_metric(
    *,
    metric: Callable[..., float],
    transform: str,
    sample_param_names: List[str] | None = None,
) -> Callable[..., Union[float, int]]:
    """Encrypted-aware analogue of :func:`fairlearn.metrics.make_derived_metric`."""
    if sample_param_names is None:
        sample_param_names = ["sample_weight"]
    return _EncryptedDerivedMetric(
        metric=metric, transform=transform, sample_param_names=sample_param_names,
    )
