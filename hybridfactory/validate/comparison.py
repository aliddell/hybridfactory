# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

from collections import OrderedDict
import itertools
from os import path as op

import numpy as np
import pandas as pd

from hybridfactory.data.annotation import load_kilosort_templates
from hybridfactory.data.dataset import AnnotatedDataSet, HybridDataSet


class PairComparison(object):
    def __init__(self, true_dataset, hybrid_dataset):
        if not isinstance(true_dataset, AnnotatedDataSet):
            raise ValueError("true_dataset must be an instance of AnnotatedDataSet")

        self._true_dataset = true_dataset

        if not isinstance(hybrid_dataset, HybridDataSet):
            raise ValueError("hybrid_dataset must be an instance of HybridDataSet")

        self._hybrid_dataset = hybrid_dataset

        rel = self._check_relationship()
        if rel != "ok":
            raise ValueError(f"relationship check failed: {rel}")

    def _check_relationship(self):
        td = self._true_dataset
        hd = self._hybrid_dataset

        if td.dtype != hd.dtype:
            return "dtype mismatch"

        if td.sample_rate != hd.sample_rate:
            return "sample_rate mismatch"

        if td.probe != hd.probe:
            return "probe mismatch"

        if not np.isin(np.unique(hd.artificial_units.true_unit), np.unique(td.annotations.cluster)).all():
            return "hybrid true_units missing from true annotations"

        return "ok"

    @property
    def dtype(self):
        return self._true_dataset.dtype

    @property
    def probe(self):
        return self._true_dataset.probe

    @property
    def sample_rate(self):
        return self._true_dataset.sample_rate


class TemplatePairComparison(PairComparison):
    def __init__(self, true_dataset, hybrid_dataset):
        super().__init__(true_dataset, hybrid_dataset)

        self._true_templates = None
        self._hybrid_templates = None

    def _check_relationship(self):
        rel = super()._check_relationship()

        if rel != "ok":
            return rel

        # template-specific checks
        td = self._true_dataset
        hd = self._hybrid_dataset

        if "template" not in td.annotations.columns:
            return "true_dataset has no template annotation"

        if "template" not in hd.annotations.columns:
            return "hybrid_dataset has no template annotation"

        return "ok"

        self._true_templates = None
        self._hybrid_templates = None

    @property
    def hybrid_templates(self):
        return self._hybrid_templates

    @property
    def true_templates(self):
        return self._true_templates

    @staticmethod
    def compare_templates(template, check_templates, how="corr"):
        """

        Parameters
        ----------
        template : numpy.ndarray
            Base template for comparison.
        check_templates : numpy.ndarray
            Templates to compare with `template`.
        how : {"corr", "norm"}, optional
            Comparison type to use (correlation (default) or Frobenius norm of difference).

        Returns
        -------
        sorted_templates : numpy.ndarray
            Templates sorted by score (best to worst match).
        scores : numpy.ndarray
            Scores computed for each template.
        indices : numpy.ndarray
            Indices of templates from best match to worst.
        """

        assert isinstance(template, np.ndarray)
        assert isinstance(check_templates, np.ndarray)
        assert how in ("corr", "norm")
        assert template.shape == check_templates[0].shape

        scores = []
        if how == "corr":
            for check_template in check_templates:
                scores.append(np.correlate(template.ravel(), check_template.ravel())[0])

            scores = np.array(scores)
            indices = np.array(list(reversed(np.argsort(scores))))
        else:
            for check_template in check_templates:
                scores.append(np.linalg.norm(template - check_template))

            scores = np.array(scores) / np.linalg.norm(template)
            indices = np.argsort(scores)

        return check_templates[indices], scores[indices], indices

    def load_templates(self, true_dir, hybrid_dir):
        if not op.isdir(true_dir):
            raise IOError(f"not a directory: {true_dir}")

        if not op.isdir(hybrid_dir):
            raise IOError(f"not a directory: {hybrid_dir}")

        if op.abspath(true_dir) == op.abspath(hybrid_dir):
            raise ValueError("true_dir and hybrid_dir can't be the same")

        self._true_templates = load_kilosort_templates(true_dir)
        self._hybrid_templates = load_kilosort_templates(hybrid_dir)

    def score_hybrid_output(self, search_radius_ms=0.5, how="corr"):
        """

        Parameters
        ----------
        search_radius_ms : float, optional
            Radius (in milliseconds) around event in which to search for matches.
        how : {"corr" (default), "norm"}, optional
            Comparison type to use (correlation or Frobenius norm of difference).

        Returns
        -------

        """
        td = self._true_dataset
        hd = self._hybrid_dataset

        true_labels = hd.artificial_units.true_unit.values
        true_times = hd.artificial_units.timestep.values

        source_labels = td.annotations.cluster.values
        hybrid_event_templates = hd.annotations.template.values
        hybrid_times = hd.annotations.timestep.values

        all_scores = {}
        # precompute all pairwise template scores
        for true_unit in hd.artificial_units.true_unit.unique():
            center_channels = hd.artificial_units[hd.artificial_units.true_unit==true_unit].center_channel.unique()
            source_event_templates = td.annotations[td.annotations.cluster == true_unit].template

            for et, cc in itertools.product(source_event_templates.unique(), center_channels):
                template = self._true_templates[et, :, :]
                shifted_template = self.shift_template(template, self.probe.channel_map[cc])

                _, t_scores, indices = self.compare_templates(shifted_template, self._hybrid_templates, how)
                for index, score in zip(indices, t_scores):
                    all_scores[(et, index)] = score

        # find (only once, since expensive) the template IDs corresponding to the true labels
        labels_templates = {}
        for label in hd.artificial_units.true_unit.unique():
            labels_templates[label] = np.unique(source_event_templates[np.where(source_labels == label)[0]])

        event_scores = np.zeros_like(true_times, dtype=np.float64)
        event_jitters = np.zeros_like(true_times)
        best_templates = np.zeros_like(true_times)

        search_radius = self.sample_rate // int(1000 * search_radius_ms)

        for k, t in enumerate(true_times):
            left, right = np.searchsorted(hybrid_times, (t - search_radius, t + search_radius))
            window_template_ids = np.unique(hybrid_event_templates[np.arange(left, right + 1)])

            template_ids = labels_templates[true_labels[k]]

            tid_scores = []
            tid_indices = []
            k_jitters = []

            for tid in template_ids:
                template_pairs = list(set(itertools.product([tid], window_template_ids)))
                template_scores = [all_scores[p] for p in template_pairs]

                if how == "corr":
                    best_match = template_pairs[np.argmax(template_scores)][1]
                else:
                    best_match = template_pairs[np.argmin(template_scores)][1]

                time_matches = np.where(window_template_ids == best_match)[0]
                delta_t = (np.abs(t - hybrid_times[np.arange(left, right + 1)][time_matches])).min()

                tid_indices.append(best_match)
                tid_scores.append(all_scores[(tid, best_match)])
                k_jitters.append(delta_t)

            if how == "corr":
                index = np.argmax(tid_scores)
            else:
                index = np.argmin(tid_scores)

            best_templates[k] = tid_indices[index]
            event_jitters[k] = k_jitters[index]
            event_scores[k] = tid_scores[index]

        return best_templates, event_jitters, event_scores

    @staticmethod
    def shift_template(template, shifted_center, percentile=80):
        """

        Parameters
        ----------
        template : numpy.ndarray
            KiloSort-style template.
        shifted_center : int
            Center channel of shifted template
        percentile : float, optional
            Percentile threshold to select channels to shift.

        Returns
        -------
        shifted_template : numpy.ndarray
            Template shifted by +/-`params.channel_shift`
        """

        assert isinstance(template, np.ndarray)
        assert 0 < percentile <= 100

        abstemplate = np.abs(template)

        tolexp = np.percentile(np.log10(abstemplate[abstemplate > 0]), percentile)
        tol = 10 ** tolexp

        channels = np.nonzero(np.any(abstemplate > tol, axis=0))[0]
        center = channels[template[:, channels].min(axis=0).argmin()]
        cmin, cmax = channels.min(), channels.max()

        # shift out of bounds?
        left, right = shifted_center - (center - cmin), shifted_center + (cmax - center)
        if left < 0 or right >= template.shape[1]:
            raise ValueError("shifted_center places template out of bounds")

        shifted_template = np.zeros_like(template)
        shifted_template[:, channels - center + shifted_center] = template[:, channels]

        return shifted_template


def build_confusion_matrix(true_labels, hybrid_labels, sort=True):
    """

    Parameters
    ----------
    true_labels : numpy.ndarray
        True labels (from curated sorting) of hybrid units.
    hybrid_labels : numpy.ndarray
        "Best match" labels from a hybrid sorting.
    sort : bool, optional
        Sort the output by best match (naively, for now).

    Returns
    -------
    confusion_matrix : pandas.DataFrame

    """

    assert isinstance(true_labels, np.ndarray)
    assert isinstance(hybrid_labels, np.ndarray)
    assert true_labels.size == hybrid_labels.size

    true_unique_labels = np.unique(true_labels)
    hybrid_unique_labels = np.unique(hybrid_labels)

    confusion_matrix = np.zeros((true_unique_labels.size, hybrid_unique_labels.size), dtype=np.int64)

    for i, true_label in enumerate(true_unique_labels):
        mask = true_labels == true_label
        hybrid_matches = hybrid_labels[mask]

        for j, hybrid_label in enumerate(hybrid_unique_labels):
            confusion_matrix[i, j] = np.count_nonzero(hybrid_matches == hybrid_label)

    if sort:
        for i in range(true_unique_labels.size):
            row = confusion_matrix[i, i:]
            sort_indices = np.flipud(np.argsort(row))

            confusion_matrix[:, i:] = confusion_matrix[:, i + sort_indices]
            hybrid_unique_labels[i:] = hybrid_unique_labels[i + sort_indices]

    # eliminate columns where all rows are zero
    keep = ~np.all(confusion_matrix == 0, axis=0)
    confusion_matrix = confusion_matrix[:, keep]
    hybrid_unique_labels = hybrid_unique_labels[keep]

    confusion_matrix = pd.DataFrame(confusion_matrix, index=true_unique_labels, columns=hybrid_unique_labels)

    confusion_matrix.columns.name = "hybrid label"
    confusion_matrix.index.name = "true label"

    return confusion_matrix


def new_template_pair_comparison(true_dataset, hybrid_dataset, firing_matrix):
    assert isinstance(true_dataset, AnnotatedDataSet)
    assert isinstance(hybrid_dataset, AnnotatedDataSet)

    # don't try this at home
    hybrid_dataset.__class__ = HybridDataSet
    hybrid_dataset._artificial_units = pd.DataFrame(OrderedDict([("timestep", firing_matrix[1, :]),
                                                                 ("true_unit", firing_matrix[2, :]),
                                                                 ("center_channel", firing_matrix[0, :])]),
                                                    dtype=np.int64)

    return TemplatePairComparison(true_dataset, hybrid_dataset)
