# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import itertools

import numpy as np


def shift_template(template, params, probe, inverse_channel_map, percentile=98.5):
    """

    Parameters
    ----------
    template : numpy.ndarray
        KiloSort-style template.
    params : module
        Session parameters.
    probe : module
        Probe parameters.
    inverse_channel_map : numpy.ndarray
        Map taking channel-sorted data back to its in-file representation.
    percentile : float, optional
        Percentile threshold to select channels to shift.

    Returns
    -------
    shifted_template : numpy.ndarray
        Template shifted by +/-`params.channel_shift`
    """

    assert isinstance(template, np.ndarray)
    assert isinstance(inverse_channel_map, np.ndarray)
    assert 0 < percentile <= 100

    abstemplate = np.abs(template)

    tolexp = np.percentile(np.log10(abstemplate[abstemplate > 0]), percentile)
    tol = 10**tolexp

    channels = np.nonzero(np.any(abstemplate > tol, axis=0))[0]

    shifted_template = np.zeros_like(template)

    # make sure our shifted channels fall in the range [0, probe.channel_map)
    if inverse_channel_map[channels].max() < probe.channel_map.size - params.channel_shift:
        shifted_template[:, params.channel_shift:] = template[:, :-params.channel_shift]
    else:
        shifted_template[:, :-params.channel_shift] = template[:, params.channel_shift:]

    return shifted_template


def compare_templates(template, check_templates, how="corr"):
    """

    Parameters
    ----------
    template : numpy.ndarray
        Base template for comparison.
    check_templates : numpy.ndarray
        Templates to compare with `template`.
    how : {"corr" (default), "norm"}, optional
        Comparison type to use (correlation or Frobenius norm of difference).

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

        scores = np.array(scores)/np.linalg.norm(template)
        indices = np.argsort(scores)

    return check_templates[indices], scores[indices], indices


def pairwise_scores(source_templates, hybrid_templates, params, probe, true_event_templates=None, how="corr"):
    """

    Parameters
    ----------
    source_templates :
        Original (i.e., unshifted) templates from source data.
    hybrid_templates : numpy.ndarray
        Templates from sorting of hybrid data.
    params : module
        Session parameters.
    probe : module
        Probe parameters.
    true_event_templates : numpy.ndarray, optional
        Indices into `true_templates` to compare a subset only.
    how : {"corr" (default), "norm"}, optional
        Comparison type to use (correlation or Frobenius norm of difference).

    Returns
    -------
    scores : dict
    """

    if true_event_templates is None:
        true_event_templates = np.arange(source_templates.shape[0])
    else:
        assert true_event_templates.min() >= 0 and true_event_templates.max() <= source_templates.shape[0]

    # inverse_channel_map[probe.channel_map] == [1, 2, ..., probe.channel_map.size - 1]
    inverse_channel_map = np.zeros(probe.channel_map.size, dtype=np.int64)
    inverse_channel_map[probe.channel_map] = np.arange(probe.channel_map.size)

    scores = {}

    for tid in true_event_templates:
        template = source_templates[tid]
        shifted_template = shift_template(template, params, probe, inverse_channel_map)

        matches, t_scores, indices = compare_templates(shifted_template, hybrid_templates, how)
        for index, score in zip(indices, t_scores):
            scores[(tid, index)] = score

    return scores


def score_hybrid_output(true_labels, true_times, source_labels, source_event_templates, source_templates,
                        hybrid_event_templates, hybrid_templates, hybrid_times, params, probe, search_radius_ms=0.5,
                        how="corr"):
    """

    Parameters
    ----------
    true_labels : numpy.ndarray
        Labels (from original sorting) of hybrid events.
    true_times : numpy.ndarray
        Times at which hybrid events have been inserted.
    source_labels : numpy.ndarray
        Cluster labels from original (curated) sorting.
    source_event_templates : numpy.ndarray
        Template IDs for each event detected in original sorting.
    source_templates : numpy.ndarray
        Templates from sorting of original data.
    hybrid_event_templates : numpy.ndarray
        Template IDs for each event detected in hybrid sorting.
    hybrid_templates : numpy.ndarray
        Templates from sorting of hybrid data.
    hybrid_times : numpy.ndarray
        Times at which sorter has detected hybrid events.
    params : module
        Session parameters.
    probe : module
        Probe parameters.
    search_radius_ms : float, optional
        Radius (in milliseconds) around event in which to search for matches.
    how : {"corr" (default), "norm"}, optional
        Comparison type to use (correlation or Frobenius norm of difference).

    Returns
    -------
    best_templates : numpy.ndarray
        "Best match" template for each hybrid event.
    event_jitters : numpy.ndarray
        Distance (in time) of best-match event from known event insertion time.
    event_scores : numpy.ndarray
        Correlation or normalized 2-norm distance of known template and best-match template.
    """

    # event templates corresponding to hybrid units
    true_event_templates = np.unique(source_event_templates[np.isin(source_labels, np.unique(true_labels))])
    # precompute all pairwise template scores
    all_scores = pairwise_scores(source_templates, hybrid_templates, params, probe, true_event_templates, how)

    # find (only once, since expensive) the template IDs corresponding to the true labels
    labels_templates = {}
    for label in np.unique(true_labels):
        labels_templates[label] = np.unique(source_event_templates[np.where(source_labels == label)[0]])

    event_scores = np.zeros(true_times.size)
    event_jitters = np.zeros_like(true_times)
    best_templates = np.zeros_like(true_times)

    search_radius = params.sample_rate // int(1000 * search_radius_ms)

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

