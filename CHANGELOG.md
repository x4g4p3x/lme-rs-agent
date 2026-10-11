# Changelog

## 0.2.1 - 2026-10-11

- Pin the published lme-rs 0.2.7 engine and report that exact version in provenance.
- Include the engine's multi-block solve, Type III hypothesis, and scalar KR fixes.


## 0.2.0 - 2026-10-10

- Retain exact CSV bytes and parsed frames for fitted models; expose `dataset_snapshot`.
- Original-data inference, prediction, bootstrap, and CV use captured frames.
- Reject comparisons of fits whose captured dataset bytes differ at the same path.
- Limit CSV snapshots to 256 MiB; preserve existing data-root checks.

## 0.1.0 - 2026-10-10

- Extract the protocol-neutral fit/session API from lme-rs-mcp.
- Pin published lme-rs 0.2.6 and use formula-based LM fitting with standard errors,
  factor metadata, offsets, and residual degrees of freedom.
- Add marginal-means and pairwise-comparison contracts with interaction grids,
  conditioning, numeric references, weighting, and multiplicity adjustments.
- Validate marginal inference against independently generated R emmeans/lmerTest
  references on deterministic synthetic unbalanced data.
- Bound bootstrap/CV engine workers and request sizes; default parallel workers to one.
- Keep statistical implementations in lme-rs and transport types in adapters.
