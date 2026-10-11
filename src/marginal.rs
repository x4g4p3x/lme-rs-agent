//! Transport-neutral contracts for engine reference-grid inference.

use crate::{AgentApiError, CachedFit, ModelKind};
use lme_rs::{DdfMethod, GridWeights, McpAdjust, ReferenceGrid};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MarginalDdf {
    /// Residual for LM, Satterthwaite for LMM.
    #[default]
    Auto,
    Asymptotic,
    Residual,
    Satterthwaite,
    KenwardRoger,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MarginalWeights {
    #[default]
    Equal,
    Proportional,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum PairwiseAdjustment {
    None,
    Bonferroni,
    #[default]
    Holm,
    Tukey,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MarginalMeansRequest {
    pub fit_id: String,
    /// Categorical fixed-effect factors defining the target cells.
    pub terms: Vec<String>,
    /// Conditioning factors; pairwise adjustments are separate within each stratum.
    #[serde(default)]
    pub by: Vec<String>,
    /// Numeric fixed-effect covariates held at these values; others use sample means.
    #[serde(default)]
    pub at: BTreeMap<String, f64>,
    #[serde(default)]
    pub weights: MarginalWeights,
    #[serde(default)]
    pub ddf_method: MarginalDdf,
    #[serde(default = "default_level")]
    pub level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PairwiseComparisonsRequest {
    #[serde(flatten)]
    pub grid: MarginalMeansRequest,
    #[serde(default)]
    pub adjust: PairwiseAdjustment,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MarginalMeanRow {
    pub cell: BTreeMap<String, String>,
    pub estimate: f64,
    pub std_error: f64,
    /// Null for asymptotic z inference (infinite df); never a JSON infinity.
    pub den_df: Option<f64>,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MarginalMeansSummary {
    pub fit_id: String,
    pub model_kind: ModelKind,
    /// Inference is on the linear predictor scale, including for GLMMs.
    pub scale: String,
    pub ddf_method: MarginalDdf,
    pub level: f64,
    pub weights: MarginalWeights,
    pub rows: Vec<MarginalMeanRow>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PairwiseComparisonRow {
    pub comparison: String,
    pub group: BTreeMap<String, String>,
    pub estimate: f64,
    pub std_error: f64,
    pub statistic: Option<f64>,
    pub den_df: Option<f64>,
    pub p_value: Option<f64>,
    pub p_adjusted: Option<f64>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PairwiseComparisonsSummary {
    pub fit_id: String,
    pub scale: String,
    pub ddf_method: MarginalDdf,
    pub adjust: PairwiseAdjustment,
    pub rows: Vec<PairwiseComparisonRow>,
}

fn default_level() -> f64 {
    0.95
}
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
fn invalid(message: &str) -> AgentApiError {
    AgentApiError::InvalidInput(message.into())
}

fn prepare(
    cached: &mut CachedFit,
    request: &MarginalMeansRequest,
) -> Result<
    (
        polars::prelude::DataFrame,
        ReferenceGrid,
        MarginalDdf,
        Option<DdfMethod>,
    ),
    AgentApiError,
> {
    if !matches!(cached.model_kind, ModelKind::Lm | ModelKind::Lmm) {
        return Err(invalid("lme-rs 0.2.7 marginal inference supports LM and LMM only; GLMM and NLMM grids are unsupported"));
    }
    if !cached.fit.converged.unwrap_or(false) {
        return Err(invalid("marginal inference requires a converged fit"));
    }
    if !request.level.is_finite() || !(0.0..1.0).contains(&request.level) || request.level == 0.0 {
        return Err(invalid(
            "level must be finite and strictly between zero and one",
        ));
    }
    if request.terms.is_empty()
        || request
            .terms
            .iter()
            .chain(&request.by)
            .any(|name| name.trim().is_empty())
    {
        return Err(invalid("terms must contain nonempty factor names"));
    }
    let mut names = std::collections::HashSet::new();
    if request
        .terms
        .iter()
        .chain(&request.by)
        .any(|name| !names.insert(name))
    {
        return Err(invalid("terms and by must be distinct without duplicates"));
    }
    if request.at.values().any(|value| !value.is_finite()) {
        return Err(invalid("reference values must be finite"));
    }
    let cell_count = request
        .terms
        .iter()
        .chain(&request.by)
        .try_fold(1usize, |count, name| {
            let levels = cached
                .fit
                .categorical_levels
                .as_ref()
                .and_then(|factors| factors.get(name))
                .ok_or_else(|| invalid("terms and by must name fitted categorical factors"))?;
            count
                .checked_mul(levels.len())
                .filter(|size| *size <= 128)
                .ok_or_else(|| invalid("at most 128 target/conditioning cells are supported"))
        })?;
    if cell_count == 0 {
        return Err(invalid("factors must have stored levels"));
    }
    let method = match request.ddf_method {
        MarginalDdf::Auto => match cached.model_kind {
            ModelKind::Lm => MarginalDdf::Residual,
            ModelKind::Lmm => MarginalDdf::Satterthwaite,
            _ => MarginalDdf::Asymptotic,
        },
        method => method,
    };
    let ddf = match method {
        MarginalDdf::Residual if cached.model_kind == ModelKind::Lm => Some(DdfMethod::Residual),
        MarginalDdf::Satterthwaite if cached.model_kind == ModelKind::Lmm => {
            Some(DdfMethod::Satterthwaite)
        }
        MarginalDdf::KenwardRoger if cached.model_kind == ModelKind::Lmm => {
            Some(DdfMethod::KenwardRoger)
        }
        MarginalDdf::Asymptotic => None,
        _ => return Err(invalid("ddf_method is incompatible with this model kind")),
    };
    let data = cached.data.clone();
    let grid = ReferenceGrid {
        terms: request.terms.clone(),
        by: request.by.clone(),
        at: request.at.clone(),
        weights: match request.weights {
            MarginalWeights::Equal => GridWeights::Equal,
            MarginalWeights::Proportional => GridWeights::Proportional,
        },
    };
    match ddf {
        Some(DdfMethod::Satterthwaite) => {
            cached
                .fit
                .with_satterthwaite(&data)
                .map_err(|e| AgentApiError::Computation(e.to_string()))?;
        }
        Some(DdfMethod::KenwardRoger) => {
            cached
                .fit
                .with_kenward_roger(&data)
                .map_err(|e| AgentApiError::Computation(e.to_string()))?;
        }
        _ => {}
    }
    Ok((data, grid, method, ddf))
}

pub(crate) fn means(
    mut cached: CachedFit,
    request: MarginalMeansRequest,
) -> Result<MarginalMeansSummary, AgentApiError> {
    let (data, grid, method, ddf) = prepare(&mut cached, &request)?;
    let result = cached
        .fit
        .emmeans_with_grid(&data, &grid, request.level, ddf)
        .map_err(|e| AgentApiError::InvalidInput(e.to_string()))?;
    if result
        .estimate
        .iter()
        .chain(result.std_error.iter())
        .chain(result.lower.iter())
        .chain(result.upper.iter())
        .any(|value| !value.is_finite())
    {
        return Err(AgentApiError::Computation(
            "non-finite marginal estimate or uncertainty".into(),
        ));
    }
    let rows = result
        .cells
        .into_iter()
        .enumerate()
        .map(|(i, cell)| MarginalMeanRow {
            cell,
            estimate: result.estimate[i],
            std_error: result.std_error[i],
            den_df: finite(result.den_df[i]),
            lower: result.lower[i],
            upper: result.upper[i],
        })
        .collect();
    Ok(MarginalMeansSummary {
        fit_id: request.fit_id,
        model_kind: cached.model_kind,
        scale: "link".into(),
        ddf_method: method,
        level: request.level,
        weights: request.weights,
        rows,
    })
}

pub(crate) fn pairs(
    mut cached: CachedFit,
    request: PairwiseComparisonsRequest,
) -> Result<PairwiseComparisonsSummary, AgentApiError> {
    let (data, grid, method, ddf) = prepare(&mut cached, &request.grid)?;
    let adjust = match request.adjust {
        PairwiseAdjustment::None => McpAdjust::None,
        PairwiseAdjustment::Bonferroni => McpAdjust::Bonferroni,
        PairwiseAdjustment::Holm => McpAdjust::Holm,
        PairwiseAdjustment::Tukey => McpAdjust::Tukey,
    };
    let result = cached
        .fit
        .emmeans_pairs_with_grid(&data, &grid, adjust, ddf)
        .map_err(|e| AgentApiError::InvalidInput(e.to_string()))?;
    if result
        .estimate
        .iter()
        .chain(result.std_error.iter())
        .any(|value| !value.is_finite())
    {
        return Err(AgentApiError::Computation(
            "non-finite pairwise estimate or uncertainty".into(),
        ));
    }
    let rows = result
        .comparisons
        .into_iter()
        .enumerate()
        .map(|(i, comparison)| PairwiseComparisonRow {
            comparison,
            group: result.groups[i].clone(),
            estimate: result.estimate[i],
            std_error: result.std_error[i],
            statistic: finite(result.statistic_values[i]),
            den_df: finite(result.den_df[i]),
            p_value: finite(result.p_value[i]),
            p_adjusted: finite(result.p_adjust[i]),
        })
        .collect();
    Ok(PairwiseComparisonsSummary {
        fit_id: request.grid.fit_id,
        scale: "link".into(),
        ddf_method: method,
        adjust: request.adjust,
        rows,
    })
}
