use lme_rs_agent::{FitModelRequest, LmeAgentApi, ModelKind, ENGINE_VERSION};

fn sleepstudy_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sleepstudy.csv")
        .display()
        .to_string()
}

#[test]
fn neutral_core_owns_fit_session_lifecycle() {
    let api = LmeAgentApi::new();
    let fit = api
        .fit_model(FitModelRequest {
            model_kind: ModelKind::Lmm,
            formula: "Reaction ~ Days + (1 | Subject)".to_string(),
            data_path: sleepstudy_path(),
            reml: Some(true),
            family: None,
            link: None,
            n_agq: None,
            start: None,
        })
        .expect("fit Sleepstudy LMM");

    assert_eq!(fit.num_obs, 180);
    assert!(fit.converged);
    assert_eq!(api.list_models().fits.len(), 1);
    assert_eq!(api.model_summary(&fit.fit_id).unwrap().formula, fit.formula);
    api.forget_model(&fit.fit_id).expect("forget model");
    assert!(api.list_models().fits.is_empty());
}

#[test]
fn protocol_names_and_json_schema_are_transport_independent() {
    assert_eq!(ENGINE_VERSION, "0.2.7");
    assert_eq!(serde_json::to_value(ModelKind::Lmm).unwrap(), "lmm");
    let schema = schemars::schema_for!(FitModelRequest);
    let schema_json = serde_json::to_value(schema).expect("serialize JSON schema");
    assert!(schema_json["properties"]["model_kind"]["$ref"]
        .as_str()
        .expect("ModelKind schema reference")
        .ends_with("/ModelKind"));
    assert_eq!(schema_json["properties"]["formula"]["type"], "string");
    assert_eq!(schema_json["properties"]["data_path"]["type"], "string");
}

#[test]
fn request_resource_limits_reject_excessive_work_before_io() {
    let api = LmeAgentApi::new();
    for request in [
        serde_json::json!({"fit_id":"missing","nsim":10001}),
        serde_json::json!({"fit_id":"missing","n_jobs":5}),
        serde_json::json!({"fit_id":"missing","nsim":0}),
    ] {
        let error = api
            .bootstrap_model(serde_json::from_value(request).unwrap())
            .unwrap_err();
        assert!(matches!(
            error,
            lme_rs_agent::AgentApiError::InvalidInput(_)
        ));
    }
    let error = api
        .fit_model(
            serde_json::from_value(serde_json::json!({
                "model_kind":"glmm","family":"poisson","n_agq":26,
                "formula":"y ~ x + (1 | g)","data_path":"missing.csv"
            }))
            .unwrap(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("n_agq"));
}
