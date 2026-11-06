use anyhow::{Result, ensure};
use axum::http::HeaderMap;
use oh_my_vllm_scheduler::session::{RequestIdentity, SessionId, SessionSource};
use serde_json::Value;
use std::io::Read;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Identity {
    pub cache: RequestIdentity,
    explicit: Option<String>,
    correlation: Option<String>,
}

impl Identity {
    pub(super) fn retained_bytes(&self) -> usize {
        self.cache.session.scope.len()
            + self.cache.session.value.len()
            + self.cache.namespace.len()
            + self.explicit.as_ref().map_or(0, String::len)
            + self.correlation.as_ref().map_or(0, String::len)
    }
}

pub(super) fn random_id() -> Result<String> {
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub(super) fn resolve(
    scope: &str,
    body: &Value,
    headers: &HeaderMap,
    previous: Option<&Identity>,
    responses: bool,
) -> Result<Identity> {
    let explicit = metadata_session(body)?;
    let correlation = correlation_header(headers)?;
    let namespace = body
        .get("prompt_cache_key")
        .filter(|value| !value.is_null())
        .map(|value| identifier(value.as_str(), "prompt_cache_key", true).map(str::to_owned))
        .transpose()?;
    if let Some(previous) = previous {
        consistent(&previous.explicit, &explicit, "metadata.session_id")?;
        consistent(&previous.correlation, &correlation, "X-Correlation-ID")?;
        ensure!(
            namespace
                .as_ref()
                .is_none_or(|value| value == &previous.cache.namespace),
            "prompt_cache_key conflicts with previous response"
        );
        return Ok(Identity {
            cache: previous.cache.clone(),
            explicit: explicit.or_else(|| previous.explicit.clone()),
            correlation: correlation.or_else(|| previous.correlation.clone()),
        });
    }
    let (source, value) = if let Some(value) = &explicit {
        (SessionSource::Explicit, value.clone())
    } else if let Some(value) = &correlation {
        (SessionSource::Correlation, value.clone())
    } else {
        (
            if responses {
                SessionSource::ResponsesRoot
            } else {
                SessionSource::Anonymous
            },
            random_id()?,
        )
    };
    Ok(Identity {
        cache: RequestIdentity {
            session: SessionId {
                scope: scope.to_owned(),
                source,
                value,
            },
            namespace: namespace.unwrap_or_default(),
        },
        explicit,
        correlation,
    })
}

fn metadata_session(body: &Value) -> Result<Option<String>> {
    let Some(metadata) = body.get("metadata") else {
        return Ok(None);
    };
    if metadata.is_null() {
        return Ok(None);
    }
    ensure!(metadata.is_object(), "metadata must be an object");
    metadata
        .get("session_id")
        .map(|value| identifier(value.as_str(), "metadata.session_id", false).map(str::to_owned))
        .transpose()
}

fn correlation_header(headers: &HeaderMap) -> Result<Option<String>> {
    let mut correlation = None;
    for value in headers.get_all("x-correlation-id") {
        let value = identifier(value.to_str().ok(), "X-Correlation-ID", false)?.to_owned();
        consistent(&correlation, &Some(value.clone()), "X-Correlation-ID")?;
        correlation = Some(value);
    }
    Ok(correlation)
}

fn identifier<'a>(value: Option<&'a str>, name: &str, empty: bool) -> Result<&'a str> {
    let value = value.ok_or_else(|| anyhow::anyhow!("{name} must be a string"))?;
    ensure!(
        (empty || !value.is_empty()) && value.len() <= 256 && !value.chars().any(char::is_control),
        "invalid {name}"
    );
    Ok(value)
}

fn consistent(previous: &Option<String>, current: &Option<String>, name: &str) -> Result<()> {
    ensure!(
        previous
            .as_ref()
            .zip(current.as_ref())
            .is_none_or(|(a, b)| a == b),
        "conflicting {name}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn branches_keep_root_and_namespace_with_separate_provenance() {
        let headers =
            HeaderMap::from_iter([("x-correlation-id".parse().unwrap(), "root".parse().unwrap())]);
        let root = resolve(
            "service",
            &json!({"prompt_cache_key":"ns"}),
            &headers,
            None,
            true,
        )
        .unwrap();
        let branch = resolve(
            "service",
            &json!({"metadata":{"session_id":"branch"}}),
            &HeaderMap::new(),
            Some(&root),
            true,
        )
        .unwrap();
        assert_eq!(branch.cache, root.cache);
        assert!(
            resolve(
                "service",
                &json!({"metadata":{"session_id":"other"}}),
                &HeaderMap::new(),
                Some(&branch),
                true
            )
            .is_err()
        );
        assert!(
            resolve(
                "service",
                &json!({"prompt_cache_key":"other"}),
                &HeaderMap::new(),
                Some(&branch),
                true
            )
            .is_err()
        );
        let other = resolve(
            "service",
            &json!({"metadata":{"session_id":"branch"}}),
            &HeaderMap::new(),
            None,
            false,
        )
        .unwrap();
        assert_ne!(other.cache.session, branch.cache.session);
    }

    #[test]
    fn identities_reject_same_source_conflicts_and_invalid_values() {
        let mut headers = HeaderMap::new();
        headers.append("x-correlation-id", "one".parse().unwrap());
        headers.append("x-correlation-id", "two".parse().unwrap());
        assert!(resolve("s", &json!({}), &headers, None, false).is_err());
        for value in [
            json!(""),
            json!(5),
            json!("line\nbreak"),
            json!("x".repeat(257)),
        ] {
            assert!(
                resolve(
                    "s",
                    &json!({"metadata":{"session_id":value}}),
                    &HeaderMap::new(),
                    None,
                    false
                )
                .is_err()
            );
        }
        let a = resolve("s", &json!({}), &HeaderMap::new(), None, false).unwrap();
        let b = resolve("s", &json!({}), &HeaderMap::new(), None, false).unwrap();
        assert_ne!(a.cache.session, b.cache.session);
    }
}
