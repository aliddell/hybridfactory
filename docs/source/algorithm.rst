Algorithm
---------

Definitions
~~~~~~~~~~~

- A *recording* :math:`R` consists of some number :math:`N` of samples collected
  from some number :math:`M` of channels (i.e., sites) and can be thought of as
  an :math:`M \times N` matrix of real numbers.
  **(In other words, your raw data file.)**
- An *event* :math:`u_j` is a set of :math:`n < N` contiguous samples across
  a collection :math:`c_j \subseteq \{ 1, \ldots, M \}` of channels, centered
  around a time :math:`t_j` at which the event is said to *occur*, with
  :math:`|c_j| = m_j \le M`.
  An event can be thought of as an :math:`m_j \times n` submatrix of :math:`R`,
  not necessarily contiguous across rows.
  An event may correspond to a true spike or a false positive (type I error).
  Let :math:`J < N` be the number of events detected by a spike sorting program.
  **(In other words, a spike, or a false spike, called by your spike
  sorter.)**
- A *sorting* is a partitioning of the set :math:`\{ 1, \ldots, J \}`
  into :math:`K` indexing sets :math:`\alpha_k`, such that
  :math:`\alpha_{a} \cap \alpha_{b} = \emptyset` if and only if :math:`a \ne b`,
  and :math:`\cup_K \alpha_k = \{1, ..., J \}`.
  **(In other words, a clustering.)**
- A *unit* :math:`U_k` is a collection of events :math:`u_{\ell}` for
  :math:`\ell \in \alpha_k`, ideally corresponding to a single neuron.
  :math:`U` is said to occur on the :math:`m` channels
  :math:`\cup_{\ell \in \alpha_k} c_{\ell}`, i.e., the set of all
  channels on which some event in :math:`U_k` occurs.
  **(In other words, a cluster.)**

Overview
~~~~~~~~

Our approach was adapted from
`this approach <https://github.com/cortex-lab/groundTruth>`__ to generating
hybrid ground-truth data.

1. For each unit :math:`U_k` consisting of :math:`p` events, form the
   :math:`m \times n \times p` array of events in :math:`U_k` such that
   the rows correspond to the channels, the columns correspond to the samples, and
   the third dimension corresponds to the events themselves.

2. Find and remove any linear trends in the time series data (i.e., the rows) in
   order to eliminate the effects of these trends on the overall variation of the
   data.

3. Using `finite differences`_, approximate the derivative (with respect to time)
   on each channel.
   This gives us an :math:`m \times (n-1) \times p` array of the
   derivatives of each event with respect to time.
   Combined with the detrending above, this reduces the influence of the slower
   post-peak phase and allows us to take fewer singular values in the
   reconstruction.

4. Reshape this array to an :math:`m(n-1) \times p` matrix :math:`A`
   such that the :math:`j`-th column of :math:`A` is the flattening of
   the :math:`j`-th event into a column vector, in whatever manner you like so long
   as you're consistent.

5. Compute the `singular value decomposition`_ of :math:`A` and reconstruct a
   low-rank approximation, :math:`\hat{A}`, of :math:`A` in order to remove noise
   from the resulting waveforms.
   (See `this post`_ for a good overview of this technique.)

6. Reshape :math:`\hat{A}` into an :math:`m \times (n - 1) \times p` array, i.e.,
   the inverse operation of (4).

7. Using cumulative sums, approximate integration (over time) of the rows of
   the reshaped :math:`\hat{A}`, i.e., the inverse operation of (3).
   This yields an :math:`m \times n \times p` array.

8. Optionally, scale the data array by some factor not too far from 1.

9. Insert the reconstructed events into the raw data at shifted times and
   channel sets.
   The shifts in the channel sets preserve the spatial distribution of the
   events themselves, that is, they don't generate new "shapes" of events.

10. Output an array of the shifted times so that you can see which of the hybrid
    events are detected by your spike sorter and how they are clustered.

.. _`finite differences`: http://mathworld.wolfram.com/FiniteDifference.html
.. _`singular value decomposition`: https://en.wikipedia.org/wiki/Singular_value_decomposition
.. _`this post`: https://blogs.sas.com/content/iml/2017/08/30/svd-low-rank-approximation.html
