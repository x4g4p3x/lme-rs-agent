use lme_rs_agent::*;
use serde_json::{json, Value};

fn fit(api: &LmeAgentApi, kind: ModelKind) -> FitSummary {
    api.fit_model(serde_json::from_value(json!({
        "model_kind": kind,
        "formula": if kind == ModelKind::Lm { "y ~ A * B + x" } else { "y ~ A * B + x + (1 | subject)" },
        "data_path": std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/marginal.csv").display().to_string()
    })).unwrap()).unwrap()
}

fn near(actual: f64, reference: f64, tolerance: f64) {
    assert!(
        (actual - reference).abs() <= tolerance * reference.abs().max(1.0),
        "{actual} != {reference}"
    );
}

#[test]
fn marginal_inference_matches_independent_r_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("data/marginal_reference.json")).unwrap();
    let api = LmeAgentApi::new();
    let lm = fit(&api, ModelKind::Lm);
    let lmm = fit(&api, ModelKind::Lmm);
    for case in reference["cases"].as_array().unwrap() {
        let fitted = if case["kind"] == "lm" { &lm } else { &lmm };
        let by = match case["by"].as_str() {
            Some(name) => vec![name.to_owned()],
            None => vec![],
        };
        let request: MarginalMeansRequest = serde_json::from_value(json!({
            "fit_id": fitted.fit_id, "terms": ["A"], "by": by, "at": {"x":0.25}, "weights":case["weights"]
        })).unwrap();
        let means = api.marginal_means(request.clone()).unwrap();
        assert_eq!(means.rows.len(), case["means"].as_array().unwrap().len());
        for actual in &means.rows {
            let expected = case["means"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| {
                    row["A"] == actual.cell["A"]
                        && actual.cell.get("B").is_none_or(|b| row["B"] == *b)
                })
                .unwrap();
            for (value, field) in [
                (actual.estimate, "emmean"),
                (actual.std_error, "SE"),
                (actual.den_df.unwrap(), "df"),
                (actual.lower, "lower.CL"),
                (actual.upper, "upper.CL"),
            ] {
                // Satterthwaite uses numerical variance derivatives; independently
                // bound df agreement to 0.1%, keeping other quantities at 0.03%.
                near(
                    value,
                    expected[field].as_f64().unwrap(),
                    if field == "df" { 1e-3 } else { 3e-4 },
                );
            }
        }
        if !by.is_empty() {
            let mut interaction = request.clone();
            interaction.terms = vec!["A".into(), "B".into()];
            interaction.by.clear();
            let grid = api.marginal_means(interaction.clone()).unwrap();
            assert_eq!(grid.rows.len(), 6);
            for row in &grid.rows {
                let original = means
                    .rows
                    .iter()
                    .find(|candidate| candidate.cell == row.cell)
                    .unwrap();
                near(row.estimate, original.estimate, 1e-12);
                near(row.std_error, original.std_error, 1e-12);
            }
            assert_eq!(
                api.pairwise_comparisons(PairwiseComparisonsRequest {
                    grid: interaction,
                    adjust: PairwiseAdjustment::Holm
                })
                .unwrap()
                .rows
                .len(),
                15
            );
        }
        let pairs = api
            .pairwise_comparisons(PairwiseComparisonsRequest {
                grid: request,
                adjust: PairwiseAdjustment::Holm,
            })
            .unwrap();
        for actual in &pairs.rows {
            let contrast = actual
                .comparison
                .split(" - ")
                .map(|cell| {
                    cell.split(", ")
                        .find_map(|entry| entry.strip_prefix("A="))
                        .unwrap_or(cell)
                })
                .collect::<Vec<_>>()
                .join(" - ");
            let expected = case["pairs"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| {
                    row["contrast"] == contrast
                        && actual.group.get("B").is_none_or(|b| row["B"] == *b)
                })
                .unwrap();
            near(
                actual.estimate,
                expected["estimate"].as_f64().unwrap(),
                3e-4,
            );
            near(actual.std_error, expected["SE"].as_f64().unwrap(), 3e-4);
            near(
                actual.p_adjusted.unwrap(),
                expected["p.value"].as_f64().unwrap(),
                3e-4,
            );
        }
    }
}

#[test]
fn engine_scope_rejects_glmm_and_supports_lmm_kenward_roger() {
    let api = LmeAgentApi::new();
    let lmm = fit(&api, ModelKind::Lmm);
    let kr = api
        .marginal_means(
            serde_json::from_value(json!({
                "fit_id":lmm.fit_id,"terms":["A"],"ddf_method":"kenward_roger"
            }))
            .unwrap(),
        )
        .unwrap();
    assert!(kr
        .rows
        .iter()
        .all(|row| row.den_df.is_some_and(|df| df > 0.0)));
    let glm = api.fit_model(serde_json::from_value(json!({
        "model_kind":"glmm","family":"gaussian","formula":"y ~ A + x + (1 | subject)",
        "data_path": std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/marginal.csv").display().to_string()
    })).unwrap()).unwrap();
    let request = serde_json::from_value(json!({"fit_id":glm.fit_id,"terms":["A"]})).unwrap();
    let error = api.marginal_means(request).unwrap_err();
    assert!(matches!(error, AgentApiError::InvalidInput(_)));
    assert!(error.to_string().contains("GLMM"));
}

#[test]
fn grid_validation_and_json_finiteness() {
    let api = LmeAgentApi::new();
    let fit = fit(&api, ModelKind::Lm);
    let base = json!({"fit_id":fit.fit_id,"terms":["A"],"ddf_method":"asymptotic"});
    let result = api
        .marginal_means(serde_json::from_value(base.clone()).unwrap())
        .unwrap();
    assert!(result.rows.iter().all(|row| row.den_df.is_none()));
    assert!(serde_json::to_value(result).unwrap()["rows"][0]["den_df"].is_null());
    for changes in [
        json!({"terms":[]}),
        json!({"terms":["A","A"]}),
        json!({"by":["A"]}),
        json!({"terms":["missing"]}),
        json!({"level":1.0}),
        json!({"ddf_method":"kenward_roger"}),
        json!({"at":{"missing":1.0}}),
    ] {
        let mut input = base.clone();
        input
            .as_object_mut()
            .unwrap()
            .extend(changes.as_object().unwrap().clone());
        assert!(api
            .marginal_means(serde_json::from_value(input).unwrap())
            .is_err());
    }
    for adjust in ["none", "holm", "bonferroni", "tukey"] {
        let mut input = base.clone();
        input["adjust"] = json!(adjust);
        let result = api
            .pairwise_comparisons(serde_json::from_value(input).unwrap())
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert!(result
            .rows
            .iter()
            .all(|row| row.p_adjusted.is_some_and(|p| (0.0..=1.0).contains(&p))));
    }
}
