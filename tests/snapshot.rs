use lme_rs_agent::*;
use serde_json::json;

#[test]
fn original_data_operations_survive_source_replacement_and_removal() {
    let directory = std::env::temp_dir().join(format!("lme-snapshot-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("input.csv");
    let bytes = include_bytes!("data/marginal.csv");
    std::fs::write(&path, bytes).unwrap();
    let api = LmeAgentApi::new();
    let request = |formula: &str| {
        serde_json::from_value(json!({
            "model_kind":"lmm", "formula":formula, "data_path":path, "reml":false
        }))
        .unwrap()
    };
    let original = api
        .fit_model(request("y ~ A * B + x + (1 | subject)"))
        .unwrap();
    let prediction = || {
        api.predict(serde_json::from_value(json!({"fit_id":original.fit_id})).unwrap())
            .unwrap()
    };
    let marginal = || {
        api.marginal_means(
            serde_json::from_value(json!({"fit_id":original.fit_id,"terms":["A"]})).unwrap(),
        )
        .unwrap()
    };
    let before = prediction().predictions;
    let means = marginal();
    let copy_path = directory.join("copy.csv");
    std::fs::write(&copy_path, bytes).unwrap();
    let equivalent = api.fit_model(serde_json::from_value(json!({"model_kind":"lmm","formula":"y ~ A + B + x + (1 | subject)","data_path":copy_path,"reml":false})).unwrap()).unwrap();
    assert!(api
        .compare_models(CompareModelsRequest {
            fit_id_a: original.fit_id.clone(),
            fit_id_b: equivalent.fit_id
        })
        .unwrap()
        .p_value
        .is_finite());
    std::fs::remove_file(copy_path).unwrap();
    std::fs::write(
        &path,
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .replace(",a,", ",z,"),
    )
    .unwrap();
    let changed = api
        .fit_model(request("y ~ A + B + x + (1 | subject)"))
        .unwrap();
    assert!(api
        .compare_models(CompareModelsRequest {
            fit_id_a: original.fit_id.clone(),
            fit_id_b: changed.fit_id
        })
        .is_err());
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        api.dataset_snapshot(&original.fit_id).unwrap().bytes(),
        bytes
    );
    assert_eq!(prediction().predictions, before);
    assert_eq!(marginal().rows[0].estimate, means.rows[0].estimate);
    let cv = api
        .cross_validate(
            serde_json::from_value(
                json!({"fit_id":original.fit_id,"group_col":"subject","n_splits":2,"seed":42}),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(cv.rmse.is_finite());
    let boot = api
        .bootstrap_model(
            serde_json::from_value(json!({"fit_id":original.fit_id,"nsim":3,"seed":42})).unwrap(),
        )
        .unwrap();
    assert_eq!(boot.nsim, 3);
    api.forget_model(&original.fit_id).unwrap();
    assert!(api.dataset_snapshot(&original.fit_id).is_err());
    std::fs::remove_dir(&directory).unwrap();
}
