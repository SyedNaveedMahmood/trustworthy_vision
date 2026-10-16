# Estimator mathematical audit — v2

The v1 mutual-control rank statistic cannot identify a dispersion-penalty benefit. For z=d+−d−, median(−z)=−median(z) and MAD(−z)=MAD(z), hence [median(z)−lambda MAD(z)]−[median(−z)−lambda MAD(−z)]=2median(z). Lambda cancels exactly, including even numbers of operators under the standard averaged median. A positive result cannot be attributed to this penalty.

The clean logit also cancels in a raw within-image pair difference and in every sham-corrected operator effect. For a linear mean, a shared off-target control cancels. Subtraction of a nonconstant vector of operator-specific controls before a median can change order statistics; scalar translation invariance does not imply invariance to that vector subtraction. Independent candidate controls remove the forced z−=−z+ relation, but increase measurement noise/cost and need semantic review.

Implemented S1 median sham effects, S2 median sham effects minus scaled MAD, and S3 independent candidate-control-adjusted median minus scaled MAD. All preserve signed values and count probability clamps. Oracle answer and frozen original prediction anchors are separate analyses. Annotation pairs remain oracle-region diagnostics with either answer anchor. Fixed image-grid candidates plus original-prediction anchor are separately labeled no-oracle.

See docs/08_ESTIMATOR_AMENDMENT.md for formulas, costs, eligibility, controls, limitations and the unchanged confirmatory block. Numerical/property tests and actual development comparisons will be linked as execution completes. Algebra alone does not establish that any corrected estimator improves evidence ranking.
