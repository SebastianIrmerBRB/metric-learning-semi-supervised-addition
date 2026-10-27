# SSL method faithfulness audit

**Audit date:** 2026-08-25
**Scope:** LRML, SLADE, SERAPH, Iscen et al. label propagation, and Hoffer & Ailon's semi-supervised neighbor embedding, including the canonical SSL configurations and the current Phase 1 run tree.

## Executive conclusion

The low-level equations are generally implemented carefully. The main risk is not a hidden sign error; it is that several local methods implement only one component of the named paper and replace the rest with the repository's generic deep metric-learning pipeline.

| Method | Mathematical core | Method/protocol as currently configured | Recommended scientific label |
|---|---|---|---|
| LRML | Faithful binary symmetric kNN Laplacian energy, including the normalized-Laplacian form | Adapted: dynamic embedding graph, arbitrary supervised DML loss, no original linear PSD/log-det problem, and Weston's repulsion is off | **LRML-regularized DML** |
| SLADE | Mostly faithful student-side basis, Gaussian, mining, and three-term loss equations | Substantially adapted self-training protocol; not the paper's full teacher/external-data/cross-fold experiment | **SLADE-style student objective** unless the paper lifecycle is enabled |
| SERAPH | Correct posterior sign, entropy objective, PSD factorization, and analytical sharpening kernel | Partial/inspired: the known-pair likelihood and trace term are absent in the active setup, calibration is changed, and optimization is minibatch SGD rather than the published EM/gradient-projection procedure | **SERAPH-inspired entropy regularization** |
| Iscen label propagation | Very close to the paper and public code for graph construction, diffusion, and entropy certainty | The active parameters are far from the paper, and the paper's confidence/class-weighted cross-entropy is not reproduced by the Phase 1 metric losses | **Iscen propagation + DML pseudo-label training** |
| Hoffer | Faithful reference sampling and unlabeled entropy structure | Material objective mismatch: local logits use squared distance, the source uses ordinary Euclidean distance, and the labeled distance-ratio half is replaced | **Hoffer-inspired squared-distance entropy** until corrected |

The most important source-level error is Hoffer's `-d^2` in place of `-d`. The most important integration error is that Iscen's certainty values are calculated but are ignored by the Phase 1 losses after thresholding. The largest method-level gap is SERAPH, which should not currently be described as a mostly faithful reproduction of the complete method.

## Phase 1 launch gate: do not launch the current tree

This is independent of the mathematical audit: the present Phase 1 configuration tree is not ready to run. A read-only traversal of every JSON file under `configs/phase_1/ssl_methods` found:

| Method | Run files | Invalid HPO reference | Missing SSL config | Filename/loss mismatch | Wrong `save_dir` |
|---|---:|---:|---:|---:|---:|
| Hoffer | 20 | 20 | 0 | 1 | 2 |
| Iscen | 20 | 20 | 5 | 12 | 12 |
| LRML | 15 | 15 | 5 | 5 | 5 |
| SERAPH | 20 | 20 | 0 | 5 | 6 |
| **Total** | **75** | **75** | **10** | **23** | **25** |

Specific blockers:

- Seventy-four `hparam_config` values point into a nonexistent `configs/hpo/ssl_methods/phase_1/...` tree. The remaining Hoffer In-Shop Multi-Similarity run has `hparam_config: null`. Thus, there are currently **zero valid Phase 1 TPE/HPO references** among these 75 files.
- Five Iscen/Cars files still point to missing `configs/ssl_label_spreading.json`; five LRML/In-Shop files still point to missing `configs/ssl_regularized_lrml.json`.
- Two SERAPH files use `configs/test_bed/ssl_seraph.json` instead of the canonical method config.
- Examples of copied/stale run content include `cars_cir.json` selecting ArcFace, `cars_nt.json` selecting Circle, and output directories such as `moar_tseta`, `test123`, or another method's In-Shop directory.
- There is no SLADE Phase 1 matrix at all.
- `python scripts/generate_ssl_method_configs.py --check` exits with `FileNotFoundError` on its stale source path `configs/ssl_label_spreading.json`, before it can compare generated files.

These errors can invalidate comparisons, collide output directories, or cause an immediate launch failure. The Phase 1 tree should be regenerated and validated before any TPE workers are started. The method findings below use the canonical files in `configs/ssl_methods`, not the stale copies in the Phase 1 tree.

## How the audit was performed

For each method I compared, in order:

1. the defining equations and algorithm in the cited paper;
2. the authors' public implementation where one exists;
3. the active local implementation, integration path, model output convention, and canonical SSL config;
4. the method-specific tests; and
5. small numerical checks where they can distinguish two plausible readings.

“Core faithful” below means that the local tensor expression represents the cited equation, up to an explicit constant reduction such as sum versus mean. It does **not** imply that the dataset, architecture, training stages, or full optimization protocol reproduces the paper.

The shared retrieval model L2-normalizes its outputs in [`models/retrieval_model.py`](../models/retrieval_model.py#L467). That convention matters for LRML graph geometry and is central to the Hoffer/normalized-softmax result.

## 1. LRML

### Reference target

Sources: [Weston et al., *Deep Learning via Semi-Supervised Embedding*](https://ronan.collobert.com/pub/2012_deeplearning_springer.pdf) and [Hoi, Liu, and Chang, *Semi-Supervised Distance Metric Learning for Collaborative Image Retrieval and Clustering*](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3307&context=sis_research). The SMU endpoint challenged automated retrieval, so its equations were also cross-checked against the [author-hosted ACM manuscript](https://www.ee.columbia.edu/ln/dvmm/publications/10/TOMCCAP10_ssml.pdf).

Hoi et al. define a binary graph

\[
W_{ij}=1 \quad\text{if } x_i\in N(x_j)\ \text{or}\ x_j\in N(x_i),
\]

where the neighbor lists are computed using ordinary Euclidean distance in the input representation. Their unlabeled regularizer is

\[
\frac12\sum_{ij}W_{ij}\lVert U^Tx_i-U^Tx_j\rVert^2
=\operatorname{tr}(XLX^TA),\qquad A=UU^T,
\]

with an optional symmetric normalized Laplacian. Their reported LRML setup uses a normalized 6-NN graph. The **complete** LRML problem also has attraction on similar labeled pairs, repulsion on dissimilar labeled pairs, a PSD constraint on `A`, and a log-determinant anti-collapse constraint/penalty.

Weston et al. replace the hard balancing constraint with a contrastive embedding loss. Their Algorithm 1 alternates a supervised update, a positive neighbor-pair update, and a random-example repulsion update sharing the neighbor pair's anchor. Their small-scale experiments use a fixed 10-NN graph and begin with five supervised-only epochs; those are experimental choices rather than universal LRML constants.

### What is faithful locally

The implementation starts at [`LRMLRegularizer`](../training/semi_supervised.py#L513), with graph construction in [`build_lrml_knn_graph`](../training/ssl/algorithms.py#L1215).

- The graph is the correct **binary union** of the directed kNN graph and its transpose. It excludes self-edges.
- With normalized embeddings, inner-product nearest neighbors have the same ordering as Euclidean nearest neighbors, so FAISS inner-product search is internally consistent.
- The unnormalized energy sums each undirected edge once. This equals the paper's `1/2` sum over the two directed entries of a symmetric `W`.
- Scaling endpoint `i` by `1/sqrt(d_i)` before summing edge distances gives exactly the symmetric normalized-Laplacian quadratic form. The implementation has an explicit numerical equivalence check against a materialized graph Laplacian.
- Uniform edge minibatching is an unbiased estimator of the full edge sum when `reduction="sum"`. `reduction="mean"` is the corresponding global edge average.
- The optional Weston repulsive branch uses the correct hinge on **Euclidean distance**, `max(0, m-d)^2`, and anchors each random pair on a sampled graph edge as Algorithm 1 describes.
- `contrastive_update_mode="alternating"` now preserves Algorithm 1's update boundary: supervised, neighbor attraction, and anchored random-pair repulsion receive three actual model-optimizer steps, and each later branch is freshly forwarded after the preceding update.
- The canonical config uses the Hoi et al. anchor values `n_neighbors=6` and `normalized_laplacian=true`, on labeled plus unlabeled nodes.

The focused LRML equation and sampling tests pass. An Adam integration test also records one optimizer step/two forwards in `joint` mode versus three optimizer steps/three forwards in `alternating` mode, verifies that both later forwards observe changed model weights, checks Adam's per-parameter step counter (`1` versus `3`), and observes the corresponding BatchNorm batch counter (`2` versus `3`, including the supervised forward).

### Deviations

1. **The graph is dynamic and constructed in the learned retrieval embedding.** Hoi and Weston construct `W` from the input representation and keep it fixed. The canonical local config rebuilds the graph after one warm-up epoch and then every epoch. This is a sensible deep adaptation, but it changes the objective because the neighborhood relation now co-evolves with the metric.

2. **Only the Laplacian component of original LRML is present.** The similar/dissimilar terms are delegated to whichever supervised metric loss the run selects. There is no linear Mahalanobis variable `A`, no original balance of `gamma_s` and `gamma_d`, no `log det(A)` constraint/penalty, and no matrix-inversion or SDP solver. A generic supervised DML loss is not algebraically the same objective.

3. **Weston's anti-collapse repulsion is disabled in the canonical config.** `contrastive=false` is the default. If the implementation is being justified by the Weston paper rather than only the Hoi Laplacian primitive, this omits an essential branch: Weston explicitly says the attractive term alone admits collapse. The generic supervised loss may prevent global collapse in practice, but that is a replacement, not Weston's mechanism.

4. **The default optional Weston branch is still combined in one backward pass.** `contrastive_update_mode="joint"` retains the previous single-step behavior for compatibility. The new opt-in `"alternating"` mode instead performs three real, ordered updates, so Adam moments, weight decay, and training-mode state can advance between supervised, attraction, and repulsion. It is a **batched** Weston analogue rather than an exact execution of Algorithm 1: each update averages a mini-batch of supervised examples or pairs instead of sampling one example/pair, and the learning-rate scheduler advances once after the three-update outer iteration.

5. **The normalized LRML attraction is not Weston's positive-pair branch.** With `normalized_laplacian=true`, attraction uses `||z_i/sqrt(d_i)-z_j/sqrt(d_j)||^2`, while Weston's eq. (3) uses plain `||z_i-z_j||^2`; repulsion already uses the plain embedding. Keep normalization for the Hoi LRML variant, but set it to `false` for a literal Weston contrastive-pair ablation.

6. **The active reduction is a mean rather than the paper's trace sum.** This is only a scale change for a fixed graph and can be absorbed by the regularizer weight, but it changes how a literal paper `lambda` transfers across datasets/graph sizes.

### Verdict

The LRML graph regularizer is implemented correctly and is the strongest source-faithful component in this group. The complete local training method is an **adapted deep LRML regularizer**, not the original LRML optimization and, with contrastive repulsion off, not the complete Weston method either. The alternating option removes the material optimizer-boundary discrepancy, but its mini-batched updates remain a deep-learning adaptation.

For a Phase 1 faithfulness anchor, keep `k=6` and the normalized Laplacian for the Hoi variant. If a Weston variant is claimed, enable the random repulsion branch, set `contrastive_update_mode="alternating"` and `normalized_laplacian=false`, and treat `k=10` plus a five-epoch supervised-only start as paper experimental anchors. Use fixed `supervised_weight` and `regularizer_weight`: target-ratio calibration and GradNorm are deliberately rejected in alternating mode because they assume simultaneous gradients at one parameter point. Do not merge the Hoi and Weston variants under one unexplained “LRML” result.

## 2. SLADE

### Reference target

Sources: [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Duan_SLADE_A_Self-Training_Framework_for_Distance_Metric_Learning_CVPR_2021_paper.pdf) and [supplement](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Duan_SLADE_A_Self-Training_CVPR_2021_supplemental.pdf).

The paper's student side consists of:

- teacher embeddings followed by k-means pseudo-labeling of the unlabeled pool;
- a learned basis matrix `W_a`, with representation `r=W_a f`;
- `L_Basis = L_CE + L_SD`;
- cosine similarities in basis-response space, split by pseudo-label into two global Gaussian distributions;
- Equation 5, `max(mu_minus - mu_plus + m, 0) + lambda(v_plus + v_minus)`;
- Equation 6 EMA updates with `beta` weighting the prior running value;
- Equation 7 mining at `s >= mu_plus` and `s <= mu_minus`; and
- Equation 9, labeled ranking + `lambda_1` unlabeled ranking + `lambda_2` basis loss.

The reported setup uses `beta=0.99`, `lambda_1=1`, `lambda_2=0.25`, batches of 32 labeled and 32 unlabeled images, and a short basis-only initialization. The broader protocol is also important: SwAV initialization, supervised teacher fine-tuning, external unlabeled sets (NABirds/CompCars/Fashion200k), student promotion, and a cross-fold handoff. Each fold emits 128 dimensions and the four fold embeddings are concatenated for evaluation.

### What is faithful locally

The basis implementation begins in [`SladeBasisLayer`](../training/ssl/slade.py#L398), and the full student loss is assembled in [`SladeRegularizer.compute_loss`](../training/ssl/slade.py#L1281).

- `r=W_a f`, the labeled softmax cross-entropy, cosine similarity in `r`, pseudo-positive/pseudo-negative separation, Equation 5, and Equation 6 all have the correct algebra and signs.
- All unordered within-batch unlabeled pairs are considered. Pairs are duplicated in both directions only when passed to pytorch-metric-learning, whose anchor semantics require that representation; this does not double the conceptual SLADE pair set.
- Equation 7 uses `mu_plus` and `mu_minus` as the positive and negative mining thresholds.
- The active default `unlabeled_ranking_loss="supervised"` applies the same pair-capable ranking criterion to labeled and mined-unlabeled streams, which is the paper's generic `L_rank` formulation.
- The basis term remains separately weighted, so the canonical `regularizer_weight=1` and `lambda_basis=0.25` reproduce `lambda_1` and `lambda_2`. `beta=0.99` also matches.
- During the basis warm-up, features are detached from the basis losses, so those losses update `W_a` without updating the embedding.

The focused Equation 1, basis, Gaussian/EMA, mining, and Equation 9 tests pass.

### Deviations and under-specified choices

1. **The canonical lifecycle is periodic re-clustering, not the paper's explicit teacher/student stages.** `within_fold_teacher_student` and `cross_fold_teacher_handoff` default to false. The current model is snapshotted implicitly for clustering and then continues training, with re-clustering every 25 epochs. The implementation supports more paper-like lifecycle flags, but the canonical config does not enable them.

2. **The canonical data protocol is a same-dataset 16-shot split.** That is the conventional SSL setting discussed in the supplement, not the paper's main external-unlabeled-data setting. The paper's robustness problem and cluster-count choice are therefore different. With `num_clusters=null`, local code uses the number of labeled classes; the paper's NABirds ablation was best at 400 clusters, matching the external pool's 400 species.

3. **The shipped SLADE run uses a frozen 768-dimensional projection stack, not the reported end-to-end ResNet-50/four-times-128 setup.** This is a legitimate controlled-comparison protocol, but not a paper reproduction.

4. **The “basis-only” warm-up is only basis-only inside the SLADE regularizer.** The repository's separate supervised ranking loss still updates the student during those steps. The paper says it first trains the basis vectors for a few iterations, which more naturally implies freezing the student globally for that initialization.

5. **Mining is additionally gated on `mu_plus > mu_minus`.** The paper gives the two thresholds without this guard. The addition avoids mining the same pair as both positive and negative when the distributions overlap, and avoids a pytorch-metric-learning assertion. It is a reasonable safety adaptation, but it suppresses all unlabeled ranking until the distributions separate.

6. **First observations seed the running Gaussians directly.** Equation 6 does not specify initialization. This avoids blending the first batch with placeholder zeros and is reasonable, but it is not stated in the paper.

7. **Gradient semantics through Equation 6 are not specified by the publication.** The canonical `sd_gradient_mode="damped"` differentiates literally through the EMA blend, multiplying the current batch statistic's Equation 5 gradient by `1-beta=0.01`. `"full"` preserves the same forward values while restoring a full batch-statistic gradient. With no released SLADE training code to settle this, neither can be called definitively source-correct; this should be reported as an implementation choice and ablated because it materially changes the basis-loss gradient scale.

8. **The paper does not publish Equation 1's contrastive margins.** Local defaults `m_pos=0`, `m_neg=1` are library defaults, not paper constants. The separate shipped HPO range currently permits `m_neg <= m_pos`, which produces invalid/contradictory trials.

### Verdict

The student-side SLADE equations are **mostly faithful and correctly implemented**. The default training lifecycle and experimental protocol are not a faithful reproduction of the paper's self-training system. It is defensible to compare “SLADE-style student objective under our common DML protocol,” provided that adaptation is explicit.

## 3. SERAPH

### Reference target

Sources: [Niu et al., arXiv:1206.4614](https://arxiv.org/abs/1206.4614) and the authors' [SERAPH software page](https://www.ms.k.u-tokyo.ac.jp/sugi/software.html).

In the simplified paper model,

\[
p_A(y\mid x_i,x_j)=\frac{1}{1+\exp\{y(d_A^2-\eta)\}},
\]

so, with `y=+1` denoting similar, `p(+1)=sigmoid(eta-d_A^2)`. Equation 7 maximizes:

- log likelihood on known similar/dissimilar pairs;
- negative entropy on every unknown pair, weighted by `mu`; and
- `-lambda tr(A)` for a PSD Mahalanobis metric `A`.

The analytical E-step gives `q(y) proportional to p(y)^(1+mu)`. The M-step optimizes known-pair likelihood plus expected unknown-pair log likelihood and trace regularization, followed by PSD projection. The paper initializes all unknown pairs as dissimilar before the first M-step.

The public MATLAB implementation confirms the posterior sign, unique unordered-pair counting (through division by two), trace penalty, PSD eigenspectrum projection, and global alternating loops. One code/paper discrepancy is worth noting: MATLAB uses exponent `1 + mu/t`, annealed by EM iteration `t`, whereas paper Equation 10 is the fixed `1 + mu`.

### What is faithful locally

The posterior head begins in [`SeraphPosteriorHead`](../training/ssl/seraph.py#L862), and batch pair scoring is in [`SeraphRegularizer.compute_loss`](../training/ssl/seraph.py#L2103).

- The similar-pair logit is correctly `kappa * (eta-d_A^2)`; there is no sign inversion.
- Direct binary entropy minimization is exactly the negative-entropy component of Equation 7 when descending a loss.
- EM mode constructs detached `q` by multiplying the posterior logit by the sharpening exponent, then uses binary cross-entropy as the M-step surrogate. This is the correct analytical E-step/M-step kernel.
- Every unordered unlabeled-unlabeled batch pair is counted once. With `include_labeled_unlabeled=true`, labeled-unlabeled pairs are also included as unknown weak-label pairs, as they should be.
- Learnable local metric options use a factor `A=B^TB`, which guarantees PSD without an eigenprojection.
- When enabled with a compatible metric, the nuclear trace term has the correct sign.

The focused posterior, entropy, EM, pair-block, and PSD tests pass.

### Material deviations

1. **The known-pair likelihood is absent.** The configured ArcFace/Circle/Multi-Similarity/etc. objective replaces the first term of Equation 7. Those losses may be good supervised DML losses, but they do not train the same posterior with the same `eta`, `A`, and weak pair labels. This is the largest conceptual omission.

2. **The active setup has neither a learned SERAPH metric nor trace regularization.** The canonical config uses `metric="identity"`, and `trace_weight` defaults to zero. The network projection can still change embeddings, but the paper's learned Mahalanobis variable and low-rank/hyper-sparsity pressure are absent.

3. **Posterior calibration is changed.** The paper's simplified model has a fixed `eta` and folds scale into `A`. The active config estimates `eta` from a 10% distance quantile and uses `kappa/batch_std`. These additions prevent trivial calibration after dropping the known-pair likelihood and using unit-normalized embeddings, but they make the objective invariant to batch distance scale and are not SERAPH.

4. **The transductive full-pool objective becomes a minibatch mean.** Local code estimates it from unordered pairs inside each regularizer batch. Population weighting between the unlabeled-unlabeled and labeled-unlabeled blocks is careful, but the pair set and optimization noise differ from the global paper objective.

5. **`mu` has been split into two independent controls.** `regularizer_weight` scales the unlabeled term, while `em_sharpening` sets the exponent directly. In the paper one `mu` does both, with exponent `1+mu`. The active `em_sharpening=2` matches `mu=1` only for the E-step, while fixed weight 15 plus GradNorm determines a different and changing objective balance.

6. **There is no persistent global `q` and no all-dissimilar first M-step.** Local code derives fresh targets from the current minibatch posterior and updates by ordinary SGD. It does not run each convex M-step to a stopping criterion or alternate global E/M states.

7. **The local fixed sharpening follows paper Equation 10 but not the public MATLAB annealing.** This is not an error; it is an unavoidable choice between conflicting primary references and should be documented as paper-faithful/code-deviating.

8. **The representation is deep and unit-normalized.** The paper learns a linear Mahalanobis metric on fixed input features. This is an intended deep adaptation, not an implementation bug.

### Verdict

The posterior and entropy equations are correct, but the complete local method is **SERAPH-inspired rather than mostly faithful**. A source-faithful baseline would need the posterior's known-pair likelihood, a learnable PSD metric, an active trace penalty, fixed paper-style calibration, and a genuine alternating/full-pair optimization schedule. If those are intentionally out of scope, the current name should explicitly say “SERAPH entropy regularizer.”

## 4. Label propagation (Iscen et al. and Zhou et al.)

### Reference target

Sources: [Iscen et al., *Label Propagation for Deep Semi-supervised Learning*](https://arxiv.org/abs/1904.04717), [Zhou et al., *Learning with Local and Global Consistency*](https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf), and the [LP-DeepSSL reference repository](https://github.com/ahmetius/LP-DeepSSL).

Iscen et al. L2-normalize descriptors and construct the directed affinity

\[
a_{ij}=[v_i^Tv_j]_+^\gamma
\quad\text{when }v_i\in NN_k(v_j),
\]

then use `W=A+A^T`, symmetric degree normalization, and solve

\[
(I-\alpha W_{norm})Z=Y
\]

with conjugate gradients. They row-normalize `Z`, assign argmax pseudo-labels, compute entropy certainty `omega_i`, normalize certainty so its maximum is one, compute inverse-population class weights `zeta_j`, and train for one epoch with certainty- and class-weighted cross-entropy. Propagation and weights are refreshed after every epoch.

The reported values are `k=50`, `gamma=3`, `alpha=0.99`, and at most 20 CG iterations. The public code adds a useful detail not present in paper Equation 5: each class's labeled seeds sum to one (`1 / labeled_count_for_class`). Zhou's original closed form includes a leading `(1-alpha)`; dropping that global factor, as Iscen does, leaves argmax labels and row-normalized probabilities unchanged.

### What is faithful locally

The main routine is [`iscen_label_spreading`](../training/ssl/algorithms.py#L678), with the graph construction shared through [`make_mixed_label_affinity`](../training/ssl/algorithms.py#L1369) and orchestration in [`IscenLabelSpreadingPseudoLabeler`](../training/semi_supervised.py#L2608).

- Descriptor normalization, exact cosine kNN search, positive-part clipping, exponentiation, directed graph construction, symmetrization, diagonal removal, and symmetric degree normalization match the paper.
- The sparse system and signs match Equation 10. The default implementation values are exactly `50/3/0.99/20`.
- Class-balanced source columns match the authors' public code.
- Negative CG overshoot is clamped as in the public code. Row normalization, entropy certainty, maximum normalization, and labeled confidence of one also match.
- Refreshing after every epoch matches Algorithm 1.
- Omitting unlabeled nodes in disconnected components with no seed is an additional robustness rule. Mathematically their source term and exact diffusion score are zero, so this is a safe explicit handling of an otherwise undefined argmax/entropy row.

The focused graph, normalized-affinity, seed, CG, pseudo-label, and confidence tests pass.

### Material deviations in the active experiment

1. **The canonical config overrides all four paper anchors:** `k=10`, `gamma=1`, `alpha=0.2`, and up to 2000 block-CG iterations. These are valid algorithm parameters, but `alpha=0.2` in particular describes much shallower propagation than the paper's `0.99`. A faithfulness baseline should include the exact `50/3/0.99/20` setting before searching away from it.

2. **A hard confidence threshold replaces graded participation for low-certainty examples.** The active `confidence_threshold=0.8` drops examples below the cutoff. The paper retains all pseudo-labeled examples and scales their cross-entropy continuously by `omega_i`.

3. **The Phase 1 metric losses ignore the retained confidence values.** The engine passes `sample_weights` only to losses declaring `supports_sample_weights`. Among the relevant local losses, only `MixedLabelPropagationProxyLoss` declares that support; the Phase 1 ArcFace, Circle, Multi-Similarity, NT-Xent, and ProxyAnchor choices do not. Consequently Iscen confidence is used as a binary admission filter, then discarded from the loss.

4. **The inverse class-population weight `zeta_j` is not implemented in this path.** A class-balanced batch sampler, if selected later, is not algebraically the same as weighting every labeled and pseudo-labeled cross-entropy term by the current propagated class population.

5. **The paper's classifier loss is replaced by arbitrary retrieval losses.** This is expected in a metric-learning benchmark, but it means the propagation core—not the complete LP-DeepSSL training objective—is what is being reproduced.

6. **The canonical config has no supervised-only warm-up (`warmup_epochs=0`).** Iscen et al. first train the network on labeled data and only then begin the per-epoch propagation/training loop. A pretrained frozen backbone may make this workable, but it is a different protocol.

### Verdict

The graph diffusion implementation is **faithful and correct**. The active Phase 1 use of its output is not faithful to LP-DeepSSL's learning stage. The minimal correction for a credible named baseline is: include the paper parameter point, preserve continuous certainty weights, implement current-class-population weights, and use a supervised warm-up. Otherwise report the method as an Iscen propagation pseudo-labeler feeding the repository's common DML objective.

## 5. Hoffer & Ailon

### Reference target

Sources: [Hoffer & Ailon, arXiv:1611.01449](https://arxiv.org/abs/1611.01449), the authors' [SemiSupContrast repository](https://github.com/eladhoffer/SemiSupContrast), and the current [pytorch-metric-learning NormalizedSoftmaxLoss documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss).

At every iteration the paper samples one labeled reference `z_c` uniformly from each class. It defines

\[
P_c(x)=\operatorname{softmax}_c(-\lVert F(x)-F(z_c)\rVert_2),
\]

using **ordinary Euclidean distance**, not squared distance. It applies cross-entropy to labeled query examples and entropy to unlabeled query examples, using the same reference set. Experiments use 32 labeled queries, 32 unlabeled queries, one reference per class, and equal weights for the two terms.

The authors' Torch criterion confirms the distance reading: it uses `nn.PairwiseDistance(2)`, negates the returned norm, then applies `LogSoftMax`. The reference code jointly computes NLL on labeled queries and entropy on the remaining query rows. A minor paper/code discrepancy is that the published `main.lua` entropy iterator is built from the full training dataset rather than explicitly removing the selected labeled subset; the local U-only selection follows the paper's stated partition.

### What is faithful locally

The implementation is [`HofferEntropyRegularizer`](../training/semi_supervised.py#L1591), with its logits and entropy at [`compute_loss`](../training/semi_supervised.py#L1762).

- It freshly samples a complete set of one uniformly chosen labeled reference per class on every step.
- It forms a C-way distance softmax for each unlabeled query and minimizes mean Shannon entropy.
- Gradients flow through both queries and sampled references.
- `reference_sets=1` matches the paper; larger values are a clearly separable local extension that averages independent C-way entropies rather than changing the denominator.

The local sampler and local entropy tests pass.

### Material deviations

1. **Local logits are `-distance_scale * d^2`, not `-d`.** This is a source contradiction, not a harmless notation choice: [`training/semi_supervised.py`](../training/semi_supervised.py#L1811) explicitly sums squared coordinate differences, while the paper and Torch `PairwiseDistance(2)` use the norm. No single `distance_scale` can transform `d^2` into `d` for all pairs. Squaring makes already distant references much less probable and changes gradients from a direction-normalized derivative to one proportional to distance.

2. **The labeled distance-ratio term is missing.** The paper applies `-log P_y(x_l)` using the same sampled references. Locally, the independent configured DML loss supplies supervised training, so the reference exemplars receive no labeled distance-ratio gradient and the two halves do not share one moving reference target.

3. **Embeddings are L2-normalized.** The paper/reference CNN exposes raw embeddings. Normalization bounds distances and turns the local squared-distance formulation into a cosine-softmax objective.

4. **The canonical regularizer weight is 0.5 rather than the paper's equal weights.** Because the local supervised objective and reduction are different, a literal numeric comparison is not meaningful, but it is another reason this is not Equation 6.

5. **`distance_scale` and multiple reference sets are local temperature/variance-control extensions.** They are useful, but have no source counterpart.

### Is this NormalizedSoftmaxLoss?

For the **current local implementation**, the parallel is exact at the logit level. If `z` and every reference `r_c` are unit norm,

\[
-s\lVert z-r_c\rVert^2
=-s(2-2z^Tr_c)
=2s\,z^Tr_c-2s.
\]

The class-independent `-2s` disappears inside softmax, leaving normalized-softmax logits `z^T r_c / T` with

\[
T=\frac{1}{2s}.
\]

At the canonical `distance_scale=s=1`, this is `T=0.5`. A numerical check with squared distances `[0.4, 3.6]` produced:

| Logit rule | Class probabilities |
|---|---|
| Hoffer paper/reference, `softmax(-d)` | `[0.779870, 0.220130]` |
| Local, `softmax(-d^2)` | `[0.960834, 0.039166]` |
| Normalized softmax, cosine with `T=0.5` | `[0.960834, 0.039166]` |

The maximum local-versus-normalized-softmax probability difference was `2.08e-17`; the paper-versus-local difference was `0.181`.

There is still one structural difference from pytorch-metric-learning's `NormalizedSoftmaxLoss`: that loss learns one normalized class weight/proxy per class, whereas Hoffer samples a real labeled exemplar per class on every step. Thus, the current term is best described as **normalized softmax with dynamic sampled exemplar proxies, followed by entropy on unlabeled queries**.

For the **original Hoffer formulation**, the parallel is only qualitative. On unit vectors its logits are `-sqrt(2-2 cos(theta))`, which is not an affine function of cosine and cannot be represented by a single normalized-softmax temperature. The user's normalized-softmax intuition therefore describes the local squared-distance deviation more exactly than it describes the cited method.

### Verdict

The sampler and entropy skeleton are correct, but the distance and missing labeled half are material. This implementation is **not mostly faithful to the complete Hoffer objective**. To claim that baseline, use ordinary Euclidean distances and train labeled and unlabeled query batches against the same sampled references. If squared distance is retained intentionally, rename/document it as the dynamic-exemplar normalized-softmax variant derived above.

## Empirical checks and test-suite state

### Focused equation checks

The following targeted command completed successfully:

```text
python -m unittest -q \
  tests.test_lrml.LRMLRegularizerTest \
  tests.test_lrml.LRMLContrastiveTermTest \
  tests.test_slade.SladeEquationOneTest \
  tests.test_slade.SladeBasisLayerTest \
  tests.test_slade.SladeLossTermTest \
  tests.test_slade.SladeComputeLossTest \
  tests.test_seraph.SeraphPosteriorHeadTest \
  tests.test_seraph.SeraphObjectiveTest \
  tests.test_seraph.SeraphLabeledUnlabeledPairsTest \
  tests.test_iscen_label_spreading.IscenLabelSpreadingTest \
  tests.test_hoffer.HofferEntropyRegularizerTest \
  tests.test_hoffer.HofferReferenceBatchSamplerTest
```

Result: **129 tests, all passing**.

This supports the conclusion that the implemented local equations are internally consistent. It does not prove paper faithfulness: notably, the Hoffer test asserts the local squared-distance formula, so it codifies rather than detects the source mismatch.

After adding the opt-in Weston update schedule, the following integration check
also passes:

```text
python -m unittest -q \
  tests.test_lrml.LRMLAlternatingOptimizerIntegrationTest
```

It runs the same deterministic sampled pairs through joint and alternating
training with Adam and a BatchNorm-equipped toy model. It checks the model
optimizer call count (`1` versus `3`), Adam's stored parameter step (`1` versus
`3`), forward count (`2` versus `3`, including supervised), BatchNorm batch
counter (`2` versus `3`), and that the later alternating forwards observe the
preceding updates. A 141-test focused regression run covering LRML, the epoch
trainer, contribution logging, target-ratio calibration, GradNorm, and learning-
rate scheduling also passes.

### Broader method tests

Running all five method modules with `unittest` executed 306 tests: **301 passed, 3 failed, and 2 errored**. The five non-passes were integration/config expectation problems:

- a missing SLADE HPO config, `configs/runs/slade/cars_teacher_student.json`, referenced by a shipped experiment;
- the SLADE paper-contrastive config contains margins `0/1`, while its test requires the empirically live `0.3/1.4` pair;
- the shipped SLADE HPO margin ranges overlap (`positive high=1`, `negative low=0.5`) despite a test requiring every possible pair of draws to be ordered;
- one SERAPH soft-rank test assumes the metric target while the current regularizer default targets the model projection; and
- one SERAPH validation test expects a “pure scale penalty” warning that current code no longer emits.

A repository-wide discovery run after the alternating change executed 1,396
tests and remained broadly red: **28 failures, 94 errors, and 1 skip**. The
reported non-passes were outside the alternating LRML path (among them missing
or inconsistent config inheritance, parser-global/default-order expectations,
existing SLADE/SERAPH assertions, scheduler wording, and platform-specific path
expectations). The focused affected suites above are therefore the relevant
regression signal; the full repository suite is not a clean baseline.

The environment does not have `pytest` installed, so these were run through the standard-library `unittest` runner. No full paper-scale training reproduction was attempted: it would require the paper datasets/protocols and substantial compute, and the Phase 1 configurations currently fail preflight.

## Recommended order before Phase 1 TPE

1. **Rebuild the Phase 1 configuration tree first.** Create every HPO JSON, repair canonical SSL references, force filename/loss/save-directory consistency, add SLADE, and make the generator's read-only `--check` pass. Add a preflight that resolves every referenced path and rejects duplicate `save_dir` values before scheduling.

2. **Decide whether the study compares named reproductions or controlled adaptations.** Both are defensible, but the result labels must distinguish them. The current implementations fit the latter better.

3. **Fix or rename Hoffer before spending TPE budget.** This is the clearest mathematical mismatch. A source-faithful branch needs `-torch.cdist(...)`, plus the labeled reference-softmax CE using the same references. The current `-d^2` branch is a valid dynamic-proxy normalized-softmax ablation and should be kept only under an explicit name.

4. **Make Iscen's training weights real.** Include `50/3/0.99/20` as the reference point, pass `omega_i` through the selected loss, implement `zeta_j`, and include a labeled warm-up. Searching propagation ranges while dropping the paper's weighting mechanism tests a different algorithm.

5. **Treat SERAPH as inspired unless expanding its scope.** A faithful implementation is substantially more than changing a range: it needs the known-pair posterior likelihood, learnable `A`, trace, and global/alternating state. If the deep adaptation is the intended contribution, keep its calibration remedies and name it accordingly.

6. **Choose the LRML variant explicitly.** Hoi-LRML supports the current 6-NN normalized Laplacian anchor; Weston-EmbedNN additionally needs the contrastive random-pair branch, `contrastive_update_mode="alternating"`, and an unnormalized positive-pair distance. Consider a fixed-graph ablation against the current adaptive graph.

7. **For SLADE, separate equation and lifecycle ablations.** The current student objective is credible. Compare periodic continuation against the explicit within-fold teacher/student mode, and report the `damped` versus `full` Equation 6 gradient choice. A true paper-protocol result additionally needs the external unlabeled pools and cross-fold/fold-concatenation procedure.

## Final classification

- **Correct core, adapted method:** LRML.
- **Mostly faithful core, adapted protocol:** SLADE.
- **Correct kernel, partial/inspired method:** SERAPH.
- **Faithful propagation, adapted and currently under-weighted training stage:** Iscen label propagation.
- **Faithful sampling skeleton but materially incorrect source distance and incomplete objective:** Hoffer.

The Phase 1 study can be scientifically useful after its configuration tree is repaired, but results should not currently be presented as five equally faithful reproductions. The cleanest framing is a controlled deep metric-learning benchmark of paper-derived SSL components, with the deviations above declared and source-faithful anchor configurations included where feasible.
