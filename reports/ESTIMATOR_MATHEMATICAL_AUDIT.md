# Estimator mathematical audit — executed v2

The original mutual-control construction cannot establish a dispersion-penalty improvement in positive-versus-negative ranking. The corrected estimators are implemented and tested, but the executed development comparison does not support the expensive independent-control estimator.

## Definitions and answer anchors

For fixed four-choice prompt q and fixed answer a, let p_a(x) be the softmax probability conditional on those four canonical answer-token logits. Define L_a(x)=log[p_a(x)/(1−p_a(x))], clipping p to [10^-6,1−10^-6] and counting clipped values. These are conditional-choice log odds, not unrestricted answer probabilities. Canonical token surface matters: the original Smol bare-letter likelihood was invalid; amendment10 records the space-prefixed correction and full-vocabulary checks.

The oracle diagnostic fixes a to the reference answer. The predicted-answer diagnostic fixes a to the unedited model argmax for every intervention, including edits whose argmax changes. Neither mode recomputes the anchor after editing. Annotated region pairs remain oracle-region diagnostics in either answer mode. The separate fixed-grid plus predicted-answer analysis avoids relevance annotations in candidate proposal, but its task/image cohort still comes from annotated datasets.

For operator o, candidate r, clean image x, removal R_o and region-preserving sham H_o:

```
D_o(r) = L_a(x) − L_a(R_o(x,r))                  raw removal sensitivity
h_o(r) = L_a(x) − L_a(H_o(x,r))                  sham sensitivity
d_o(r) = D_o(r) − h_o(r)
       = L_a(H_o(x,r)) − L_a(R_o(x,r))           sham-corrected sensitivity
M(v)   = median_o(v_o)
V(v)   = 1.4826 median_o |v_o − M(v)|             scaled MAD
S1(r)  = M(d(r))
S2(r)  = M(d(r)) − λ V(d(r))
z_o(r) = d_o(r) − d_o(c(r))
S3(r)  = M(z(r)) − λ V(z(r)),  λ = 0.5
```

MAD here measures disagreement between three correlated interventions. It is not a standard error or calibrated uncertainty estimate. Signed scores are preserved; negative effects are not truncated. The coefficient was fixed before intervention results and was not tuned to force improvement.

## Exact cancellations in the ranking metric

The pair outcome compares S(r+) with S(r−), with half credit for ties. The shared clean log odds cancel in each raw operator pair difference, and cancel entirely in every sham-corrected effect. Sham terms themselves generally differ between candidates and therefore can affect ranking.

If c(r+)=r− and c(r−)=r+, set v=d(r+)−d(r−). Then z(r+)=v and z(r−)=−v. Median is odd and MAD is even (including an even operator count with the averaged median), so:

```
S3(r+) − S3(r−)
 = [M(v) − λV(v)] − [M(−v) − λV(−v)]
 = 2M(v)
```

Thus every λ gives the same mutual-control pair decision. This does not imply that the decision equals the difference of the two separately computed candidate medians: median subtraction is nonlinear.

A shared scalar control subtracted from all operator effects translates median and leaves MAD unchanged, so it cancels in the pair contrast. A shared operator-specific vector cancels under linear mean aggregation, including any common additive correction after aggregation. A nonconstant shared vector subtracted before a median can change operator order and therefore ranking. That is a nonlinear aggregation effect, not evidence that the shared control contains region-specific information. Candidate-specific independent controls avoid forced antisymmetry; they do not guarantee causal irrelevance, beneficial correction or low variance.

## Implemented alternatives and inference cost

S1 uses region-preserving shams. S2 adds a signed consistency penalty. S3 uses separate, equal-size translated control rectangles per candidate. The control sampler receives only image dimensions, candidate coordinates and fixed seed; it cannot inspect relevance labels, questions, predictions or sensitivities. It avoids its own candidate, but may overlap other meaningful content. Control contamination remains a semantic-audit dependency. Geometry failure is an explicit exclusion before model outcomes.

For a two-candidate, three-operator pair: raw single operator costs3 scores including clean; raw mean/median7; S1/S2 13; S3 25. Gold versus frozen-predicted anchors reuse the same probability vectors and add no forwards. Common-output analysis does not make S3 free at deployment; direct method cost is reported separately.

## Executed tests and empirical outcome

Numerical and Hypothesis properties in `tests/test_v2_protocol.py`, `test_v2_integrity.py` and `test_v2_evaluation.py` verify sign symmetry/MAD parity, mutual-control cancellation, shared-control properties, independent-control behavior, anchor invariance, probability normalization and paired statistical alignment. The final JUnit record is `reports/evidence/v2/tests_v2.xml`; the complete suite result is recorded in STATUS, rather than inferred from this derivation.

On81 development image pairs shared across three backbones, the primary Qwen + corrected-Smol macro ranking is S1 84.54%, S2 85.76%, S3 80.23%, versus raw median85.53%. S3−raw median is−5.30pp, descriptive paired95% CI[−12.02,+1.14]. S3−S2 is−5.53pp[−10.44,−1.22]. These are development comparisons with unresolved semantic/reference validity and comparator selection on the same data; they are not confirmatory tests. The corrected penalty can affect rankings and does in the pooled run, but extra independent controls have not justified their cost. No positive conclusion follows from repairing the algebra.

Versioned protocol: [amendment08](../docs/08_ESTIMATOR_AMENDMENT.md). Executed estimates and per-cell intervals: [E3](E3_EXPLORATORY_COMPARISON.md), [full JSON](evidence/v2/E3_replication.json). The original protocol and earlier negative results remain preserved. E5 remains blocked.
