//! In-memory cache of fitted models keyed by `fit_id`.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::data::DatasetSnapshot;
use lme_rs::LmeFit;
use polars::prelude::DataFrame;

use crate::dto::ModelKind;

#[derive(Debug, Clone)]
pub struct CachedFit {
    /// Semantic family of the cached model.
    pub model_kind: ModelKind,
    pub formula: String,
    pub data_path: PathBuf,
    /// Immutable parsed input and exact source bytes used for all original-data operations.
    pub data: DataFrame,
    pub dataset: DatasetSnapshot,
    /// REML setting when applicable to the model family.
    pub reml: Option<bool>,
    /// Adaptive Gauss-Hermite quadrature points for GLMM/NLMM.
    pub n_agq: Option<usize>,
    /// User-provided NLMM starting values; `None` means self-start was requested.
    pub start: Option<BTreeMap<String, f64>>,
    /// Unified `lme-rs` fit representation used by LMM, GLMM, and NLMM models.
    pub fit: LmeFit,
}

#[derive(Debug, Default)]
pub struct FitSession {
    inner: Mutex<HashMap<String, CachedFit>>,
}

impl FitSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, fit_id: String, entry: CachedFit) {
        self.inner
            .lock()
            .expect("fit session lock poisoned")
            .insert(fit_id, entry);
    }

    pub fn get(&self, fit_id: &str) -> Option<CachedFit> {
        self.inner
            .lock()
            .expect("fit session lock poisoned")
            .get(fit_id)
            .cloned()
    }

    pub fn list(&self) -> Vec<(String, CachedFit)> {
        self.inner
            .lock()
            .expect("fit session lock poisoned")
            .iter()
            .map(|(id, c)| (id.clone(), c.clone()))
            .collect()
    }

    pub fn remove(&self, fit_id: &str) -> bool {
        self.inner
            .lock()
            .expect("fit session lock poisoned")
            .remove(fit_id)
            .is_some()
    }
}
