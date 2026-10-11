//! Protocol-neutral agent API for [`lme-rs`](https://github.com/x4g4p3x/lme-rs).
//!
//! This crate owns statistical orchestration, validation, typed DTOs, and in-process
//! fit sessions. Transport adapters such as `lme-rs-mcp` and `lme-rs-scp` depend on it;
//! it does not depend on either protocol implementation.

mod api;
mod data;
mod dto;
mod marginal;
mod session;

pub use api::{AgentApiError, LmeAgentApi};
pub use data::{load_csv, DatasetSnapshot};
pub use dto::{
    AnovaRequest, AnovaRow, AnovaSummary, BootConfintRow, BootSummary, BootstrapRequest,
    CompareModelsRequest, CompareModelsSummary, ConfidenceIntervalMethod, ConfidenceIntervalRow,
    ConfidenceIntervalsRequest, ConfidenceIntervalsSummary, CrossValidateRequest,
    CrossValidationFold, CrossValidationSummary, DiagnosticsSummary, FitIdRequest, FitListEntry,
    FitListSummary, FitLmmRequest, FitModelRequest, FitSummary, ForgetFitResult,
    ModelBootstrapRequest, ModelKind, PredictRequest, PredictionMode, PredictionScale,
    PredictionSummary,
};
pub use marginal::*;
pub use session::{CachedFit, FitSession};

/// Statistical engine package represented in provenance records.
pub const ENGINE_NAME: &str = "lme-rs";

/// Published `lme-rs` version used by this release of the agent core.
pub const ENGINE_VERSION: &str = "0.2.7";

/// Shared orchestration version represented in provenance records.
pub const AGENT_VERSION: &str = env!("CARGO_PKG_VERSION");
