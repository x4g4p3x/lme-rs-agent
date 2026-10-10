# Agent and contributor pre-flights

Protocol-neutral Rust agent core around `lme-rs`. Do not add MCP or SCP transport types
to this crate.

Before handing off Rust changes, run:

```powershell
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

Statistical algorithms belong in the sibling `lme-rs` repository. Protocol adapters
belong in `lme-rs-mcp` or `lme-rs-scp`.
