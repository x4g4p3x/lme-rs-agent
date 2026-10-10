//! Load tabular data from local CSV paths for agent-facing operations.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use polars::prelude::*;

/// Exact bytes parsed for a fitted model, retained independently of the source file.
#[derive(Debug, Clone)]
pub struct DatasetSnapshot {
    pub path: PathBuf,
    bytes: Arc<[u8]>,
}

impl DatasetSnapshot {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Resolve and load a CSV file referenced by an agent request.
///
/// When `LME_DATA_ROOT` is set, relative `data_path` values are resolved under that
/// directory before canonicalization. The canonical path must still lie under the root.
/// `LME_MCP_DATA_ROOT` is accepted as a backwards-compatible alias.
pub fn load_csv(data_path: &str) -> Result<(PathBuf, DataFrame)> {
    let (snapshot, data) = load_csv_snapshot(data_path)?;
    Ok((snapshot.path, data))
}

pub(crate) fn load_csv_snapshot(data_path: &str) -> Result<(DatasetSnapshot, DataFrame)> {
    let path = Path::new(data_path);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else if let Some((_, root)) = optional_data_root() {
        root.join(path)
    } else {
        path.to_path_buf()
    };

    let canonical = resolved
        .canonicalize()
        .with_context(|| format!("data_path not found or inaccessible: {data_path}"))?;

    if let Some((variable, root)) = optional_data_root() {
        let root_canonical = root
            .canonicalize()
            .with_context(|| format!("{variable} not found or inaccessible: {root:?}"))?;
        if !canonical.starts_with(&root_canonical) {
            anyhow::bail!(
                "data_path {:?} is outside {variable} {:?}",
                canonical,
                root_canonical
            );
        }
    }

    if canonical.extension().and_then(|s| s.to_str()) != Some("csv") {
        anyhow::bail!("only CSV files are supported by this adapter version");
    }

    let file = File::open(&canonical).with_context(|| format!("failed to open {canonical:?}"))?;
    const MAX_BYTES: u64 = 256 * 1024 * 1024;
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() as u64 <= MAX_BYTES, "CSV input exceeds 256 MiB");
    let df = CsvReader::new(Cursor::new(bytes.as_slice()))
        .finish()
        .with_context(|| format!("failed to parse CSV {canonical:?}"))?;

    Ok((
        DatasetSnapshot {
            path: canonical,
            bytes: bytes.into(),
        },
        df,
    ))
}

fn optional_data_root() -> Option<(&'static str, PathBuf)> {
    std::env::var_os("LME_DATA_ROOT")
        .map(|root| ("LME_DATA_ROOT", PathBuf::from(root)))
        .or_else(|| {
            std::env::var_os("LME_MCP_DATA_ROOT")
                .map(|root| ("LME_MCP_DATA_ROOT", PathBuf::from(root)))
        })
}
