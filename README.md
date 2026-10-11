# lme-rs-agent

Protocol-neutral statistical agent core for [lme-rs](https://github.com/x4g4p3x/lme-rs).

The crate owns `LmeAgentApi`, request/response DTOs, validation, CSV loading, and the
in-process fitted-model session. It deliberately has no MCP or SCP transport dependency.

```text
                    lme-rs
                       ^
                       |
                lme-rs-agent
               /            \
              ^              ^
       lme-rs-mcp       lme-rs-scp
```

`lme-rs-agent` is not a statistics reimplementation. Numerical work remains in `lme-rs`.
The current agent release pins the published `lme-rs 0.2.7` engine exactly so provenance
and tested behavior identify the same statistical implementation.

## Data paths

Set `LME_DATA_ROOT` to constrain relative and absolute CSV paths to a canonicalized data
directory. `LME_MCP_DATA_ROOT` remains a deprecated compatibility alias for existing MCP
installations.

Version 0.2.0 retains exact CSV bytes (up to 256 MiB) and the parsed frame for each
fit. `dataset_snapshot(fit_id)` supplies provenance bytes; original-data operations
use the captured frame even after the source changes or disappears. Prediction on
an explicitly supplied new CSV still loads that file. Forgetting a model releases
its snapshot. Consumers constructing `CachedFit` directly must supply `data` and
`dataset` fields.

## Development

The shared API includes marginal means and adjusted pairwise comparisons for
LM/LMM. `MarginalMeansRequest` carries target factors, conditioning factors,
numeric references, equal/proportional nuisance weighting, confidence level, and
model-aware df. `PairwiseComparisonsRequest` adds Holm/Bonferroni/Tukey/no adjustment.
Results use the link scale; asymptotic df and unavailable p-values use JSON null.
GLMM/NLMM marginal inference and reference grids with offsets are unsupported.

The bounded agent contract permits 1–10000 bootstrap replicates, 2–100 CV folds,
1–25 AGQ points, 1–4 explicit engine workers (default one), and up to 128
target/conditioning cells. Statistical implementations remain in lme-rs.

`tests/data/marginal_reference.json` records independent R emmeans/lmerTest results
and their generator/version provenance for deterministic synthetic data. Ordinary
Rust tests consume the checked-in reference and require no R installation. Regenerate
from this directory with `Rscript tests/generate_marginal_reference.R` after installing
emmeans, lmerTest, and jsonlite. Review numerical differences rather than automatically
replacing the reference. See CHANGELOG.md for release scope.

```powershell
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

## License

MIT.
