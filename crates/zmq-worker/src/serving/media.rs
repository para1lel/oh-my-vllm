use anyhow::{Result, bail, ensure};
use axum::http::Uri;
use serde_json::{Value, json};

pub(super) fn content(value: &Value, allow_media: bool) -> Result<Value> {
    let Value::Array(parts) = value else {
        ensure!(value.is_null() || value.is_string(), "invalid content");
        return Ok(value.clone());
    };
    let mut normalized = Vec::with_capacity(parts.len());
    let mut has_media = false;
    for part in parts {
        match part["type"].as_str() {
            Some("text" | "input_text" | "output_text") => {
                let text = part["text"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("text part requires text"))?;
                normalized.push(json!({"type": "text", "text": text}));
            }
            Some("image_url" | "input_image") if allow_media => {
                normalized.push(media_part(part, "image")?);
                has_media = true;
            }
            Some("video_url" | "input_video") if allow_media => {
                normalized.push(media_part(part, "video")?);
                has_media = true;
            }
            _ => bail!("unsupported content part"),
        }
    }
    if has_media {
        Ok(Value::Array(normalized))
    } else {
        Ok(json!(
            normalized
                .iter()
                .map(|p| p["text"].as_str().unwrap())
                .collect::<String>()
        ))
    }
}

fn media_part(part: &Value, kind: &str) -> Result<Value> {
    let field = format!("{kind}_url");
    let value = &part[&field];
    let url = value
        .as_str()
        .or_else(|| value["url"].as_str())
        .ok_or_else(|| anyhow::anyhow!("{field} requires a URL"))?;
    ensure!(!url.is_empty(), "empty media URL");
    if url.starts_with("data:") {
        let (header, data) = url
            .split_once(',')
            .ok_or_else(|| anyhow::anyhow!("invalid media data URI"))?;
        ensure!(
            header.starts_with(&format!("data:{kind}/"))
                && header.ends_with(";base64")
                && valid_base64(data),
            "invalid media data URI"
        );
    } else {
        let uri: Uri = url.parse()?;
        ensure!(
            matches!(uri.scheme_str(), Some("https" | "http"))
                && uri.host().is_some_and(|host| !host.is_empty())
                && uri
                    .authority()
                    .is_some_and(|authority| !authority.as_str().contains('@')),
            "invalid media URL"
        );
    }
    let mut options = json!({"url": url});
    if let Some(detail) = value.get("detail").or_else(|| part.get("detail")) {
        ensure!(
            kind == "image" && ["auto", "low", "high"].iter().any(|s| detail == *s),
            "invalid image detail"
        );
        options["detail"] = detail.clone();
    }
    let mut normalized = json!({"type": field});
    normalized[&field] = options;
    Ok(normalized)
}

fn valid_base64(data: &str) -> bool {
    if data.is_empty() || !data.len().is_multiple_of(4) {
        return false;
    }
    let unpadded = data.trim_end_matches('=');
    let padding = data.len() - unpadded.len();
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if padding > 2 || !unpadded.bytes().all(|b| alphabet.contains(&b)) {
        return false;
    }
    let last = unpadded
        .bytes()
        .last()
        .and_then(|b| alphabet.iter().position(|v| *v == b));
    match padding {
        0 => true,
        1 => last.is_some_and(|v| v & 3 == 0),
        _ => last.is_some_and(|v| v & 15 == 0),
    }
}

pub(super) fn count(content: &Value) -> usize {
    content.as_array().map_or(0, |parts| {
        parts
            .iter()
            .filter(|part| part["type"] == "image_url" || part["type"] == "video_url")
            .count()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_content_keeps_order_and_rejects_invalid_media() {
        let input = json!([
            {"type":"input_text","text":"before"},
            {"type":"input_image","image_url":"data:image/png;base64,AQID","detail":"low"},
            {"type":"text","text":"after"},
            {"type":"input_video","video_url":"https://example.invalid/video.mp4"}
        ]);
        let normalized = content(&input, true).unwrap();
        assert_eq!(normalized[0]["text"], "before");
        assert_eq!(normalized[1]["image_url"]["detail"], "low");
        assert_eq!(normalized[2]["text"], "after");
        assert_eq!(
            normalized[3]["video_url"]["url"],
            "https://example.invalid/video.mp4"
        );
        assert_eq!(count(&normalized), 2);
        assert!(content(&input, false).is_err());
        for url in [
            "file:///tmp/image.png",
            "https:///missing-host",
            "https://:80",
            "https://user@",
            "data:video/mp4;base64,AQID",
            "data:image/png;base64,====",
            "data:image/png;base64,AA=A",
            "data:image/png;base64,AB==",
        ] {
            assert!(content(&json!([{"type":"image_url","image_url":{"url":url}}]), true).is_err());
        }
    }
}
