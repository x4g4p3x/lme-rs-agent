# Changelog

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
