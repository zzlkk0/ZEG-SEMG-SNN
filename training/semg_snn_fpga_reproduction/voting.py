"""Deterministic, torch-free temporal voting post-process.

Implements the semantics described in DELAY_RETRAIN_TOWARD_PAPER_SPEC.md 2.1:
walk the per-subject, time-ordered prediction sequence; when a window's raw
prediction differs from the currently "accepted" label, only take the switch
if the *next* window's raw prediction agrees with it (i.e. two consecutive
windows must agree before a label change is accepted). An isolated single-
window jump that the following window does not confirm is treated as noise
and discarded -- the previous accepted label is kept for that window.

This runs entirely on the host after inference (numpy only, no torch), and
is a one-window (100 ms) lookahead filter, not a causal-only filter: the
host is expected to buffer one extra window of latency to apply it.
"""

from __future__ import annotations

import numpy as np


def paper_style_vote(
    predictions: np.ndarray, subjects: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    """Debounce isolated single-window label jumps, independently per subject.

    Args:
        predictions: [N] int array, raw per-window argmax predictions.
        subjects: [N] int array, subject id for each window.
        starts: [N] int array, window start sample index (for time ordering).

    Returns:
        [N] int array of voted predictions, same order as the inputs.
    """
    predictions = np.asarray(predictions)
    subjects = np.asarray(subjects)
    starts = np.asarray(starts)
    output = predictions.copy()

    for subject in np.unique(subjects):
        indexes = np.flatnonzero(subjects == subject)
        indexes = indexes[np.argsort(starts[indexes])]
        if len(indexes) == 0:
            continue
        sequence = predictions[indexes]
        n = len(sequence)
        voted = np.empty(n, dtype=sequence.dtype)

        accepted = int(sequence[0])
        voted[0] = accepted
        for i in range(1, n):
            current = int(sequence[i])
            if current == accepted:
                voted[i] = accepted
                continue
            # current differs from the accepted label: only take the switch
            # if the next window's raw prediction also agrees with it.
            confirmed = i + 1 < n and int(sequence[i + 1]) == current
            if confirmed:
                accepted = current
                voted[i] = accepted
            else:
                voted[i] = accepted  # isolated jump, discarded

        output[indexes] = voted

    return output


def two_window_debounce_vote(
    predictions: np.ndarray, subjects: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    """Causal alternative already used by evaluate_checkpoint.py: a switch is
    accepted once the *same* candidate label has appeared twice among the
    non-accepted predictions seen so far (no lookahead, purely backward
    looking). Included for comparison against paper_style_vote, since the
    two are similar in spirit (both require two agreeing windows before a
    switch) but differ in whether they use a lookahead window.
    """
    predictions = np.asarray(predictions)
    subjects = np.asarray(subjects)
    starts = np.asarray(starts)
    output = predictions.copy()

    for subject in np.unique(subjects):
        indexes = np.flatnonzero(subjects == subject)
        indexes = indexes[np.argsort(starts[indexes])]
        if len(indexes) == 0:
            continue
        sequence = predictions[indexes]
        accepted = int(sequence[0])
        candidate = accepted
        candidate_count = 0
        voted = np.empty(len(sequence), dtype=sequence.dtype)
        voted[0] = accepted
        for i in range(1, len(sequence)):
            current = int(sequence[i])
            if current == accepted:
                candidate, candidate_count = accepted, 0
            elif current == candidate:
                candidate_count += 1
                if candidate_count >= 2:
                    accepted, candidate_count = candidate, 0
            else:
                candidate, candidate_count = current, 1
            voted[i] = accepted
        output[indexes] = voted

    return output
