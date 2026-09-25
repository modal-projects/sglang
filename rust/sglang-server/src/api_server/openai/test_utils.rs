//! Shared HTTP test harness and `openai.rs`-level handler tests.
//!
//! Submodule tests live next to the code they cover: `chat`, `completions`,
//! `tools`, and `reasoning` each carry their own
//! `#[cfg(test)] mod tests`. This module keeps the fixtures they all share —
//! channel fixtures (`senders`, `chunk`, `submitted`, `chat_submitted`) and the
//! full-router harness (`server_args`, `app_state`,
//! `oneshot`, `post_json`, `body_json`) — plus the handler-level tests that
//! exercise [`routes`] end to end. The helpers are `pub(super)` so sibling
//! test modules can import them via `super::super::test_utils::*`.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use serde_json::json;
use tower::util::ServiceExt;

use super::{openai_error, routes};
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::wiring::Senders;

pub(super) fn senders() -> Senders {
    Senders {
        tok_manager_tx: flume::unbounded().0,
        abort_tx: flume::unbounded().0,
        tokenizer_tx: flume::unbounded().0,
        detokenizer_tx: vec![],
    }
}

pub(super) fn chunk(rid: &str, text: &str, done: bool) -> ResponseItem {
    let output = ChunkEvent {
        rid: rid.into(),
        text: text.into(),
        token_ids: vec![1],
        prompt_tokens: 5,
        completion_tokens: 1,
        finish_reason: done.then(|| {
            serde_json::from_value(serde_json::json!({
                "type": "stop",
                "matched": "</s>"
            }))
            .unwrap()
        }),
        ..Default::default()
    };
    if done {
        ResponseItem::Done(output)
    } else {
        ResponseItem::Frame(output)
    }
}

pub(super) fn terminal_cases() -> Vec<(serde_json::Value, Option<u16>, bool)> {
    vec![
        (
            json!({"type": "abort", "status_code": 500, "message": "engine failed"}),
            Some(500),
            true,
        ),
        (
            json!({"type": "abort", "status_code": 502}),
            Some(502),
            true,
        ),
        (
            json!({"type": "abort", "status_code": 599}),
            Some(599),
            true,
        ),
        (
            json!({"type": "abort", "status_code": 408, "err_type": "encoder_timeout"}),
            Some(408),
            true,
        ),
        (
            json!({"type": "stop", "matched": "invalid token", "err_type": "invalid_token"}),
            Some(500),
            true,
        ),
        (
            json!({"type": "abort", "status_code": 503}),
            Some(503),
            false,
        ),
        (
            json!({"type": "abort", "status_code": 429}),
            Some(429),
            false,
        ),
        (
            json!({"type": "abort", "status_code": 400}),
            Some(400),
            false,
        ),
        (
            json!({"type": "abort", "status_code": 400, "err_type": "cancelled"}),
            Some(400),
            false,
        ),
        (
            json!({"type": "abort", "status_code": 408}),
            Some(408),
            false,
        ),
        (json!({"type": "abort"}), None, false),
        (
            json!({"type": "stop", "matched": "NaN happened"}),
            None,
            false,
        ),
        (json!({"type": "stop", "matched": 2}), None, false),
        (json!({"type": "length", "length": 1}), None, false),
    ]
}

pub(super) fn terminal_chunk(rid: &str, reason: serde_json::Value) -> ResponseItem {
    let ResponseItem::Done(mut output) = chunk(rid, "terminal text", true) else {
        unreachable!()
    };
    let packed = rmp_serde::to_vec_named(&reason).unwrap();
    output.finish_reason = Some(rmp_serde::from_slice(&packed).unwrap());
    ResponseItem::Done(output)
}

pub(super) fn assert_terminal_frames(frames: &[String], code: Option<u16>, engine_fault: bool) {
    assert_eq!(frames.last().unwrap(), "[DONE]");
    assert_eq!(frames.iter().filter(|frame| *frame == "[DONE]").count(), 1);
    let values: Vec<serde_json::Value> = frames[..frames.len() - 1]
        .iter()
        .map(|frame| serde_json::from_str(frame).unwrap())
        .collect();
    let errors: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.get("error").is_some())
        .collect();
    if let Some(code) = code {
        assert_eq!(errors.len(), 1, "{frames:?}");
        assert_eq!(errors[0].1["error"]["code"], code);
        assert!(errors[0].1.get("choices").is_none());
        if engine_fault {
            assert_eq!(errors[0].0, values.len() - 1, "{frames:?}");
            assert!(
                values.iter().all(|value| value
                    .get("choices")
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(|choices| !choices.is_empty())),
                "{frames:?}"
            );
        }
    } else {
        assert!(errors.is_empty(), "{frames:?}");
        assert!(
            values
                .iter()
                .any(|value| !value["choices"][0]["finish_reason"].is_null())
        );
    }
    if !engine_fault {
        assert!(values.last().unwrap()["usage"].is_object(), "{frames:?}");
    }
}

/// A submitted legacy completion choice.
pub(super) fn submitted(
    index: usize,
    prompt_index: usize,
    rid: &str,
) -> (
    super::completions::SubmittedChoice,
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    (
        super::completions::SubmittedChoice {
            index,
            prompt_index,
            rid: rid.into(),
            echo: String::new(),
            rx,
        },
        tx,
    )
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes).
pub(super) fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, Rid, tokio::sync::mpsc::Receiver<ResponseItem>),
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    ((index, rid.into(), rx), tx)
}

pub(super) fn server_args() -> Arc<ServerArgs> {
    Arc::new(ServerArgs {
        served_model_name: "model".into(),
        ..Default::default()
    })
}

pub(super) fn app_state(senders: Senders) -> Arc<super::AppState> {
    Arc::new(super::AppState {
        senders,
        response_buf: 8,
        server_args: server_args(),
        chat_formatter: None,
        response_activity: Default::default(),
        startup_readiness: Default::default(),
    })
}

pub(super) fn senders_closed() -> Senders {
    // Dropping the receivers disconnects the channels; the senders stay
    // valid (moveable) but every send reports `Err`, the shutdown state
    // `submit` surfaces as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (abort_tx, abort_rx) = flume::unbounded();
    drop(abort_rx);
    let (tok_tx, tok_rx) = flume::unbounded();
    drop(tok_rx);
    Senders {
        tok_manager_tx: tm_tx,
        abort_tx,
        tokenizer_tx: tok_tx,
        detokenizer_tx: vec![],
    }
}

/// Serve one request through the full router (extractors, auth, routing).
/// `with_state` consumes the state into a `Router<()>`, which is what
/// implements `tower::Service`.
pub(super) async fn oneshot(app: Router<()>, req: Request<Body>) -> Response {
    app.oneshot(req).await.unwrap()
}

pub(super) async fn post_json(app: Router<()>, path: &str, body: serde_json::Value) -> Response {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    oneshot(app, req).await
}

pub(super) async fn body_json(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// The common StatusCode→error helper follows `error_response`'s shape:
/// unary requests get the JSON error with its status; a committed stream gets
/// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
/// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
#[tokio::test]
async fn openai_error_response_covers_unary_and_sse() {
    let unary = openai_error(StatusCode::BAD_REQUEST, "bad input", false);
    assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
    let value = body_json(unary).await;
    assert_eq!(value["error"]["message"], "bad input");
    assert_eq!(value["error"]["type"], "BadRequestError");
    assert_eq!(value["error"]["code"], 400);
    assert!(value["error"]["param"].is_null());

    let streamed = openai_error(StatusCode::BAD_REQUEST, "bad input", true);
    assert_eq!(streamed.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
    assert_eq!(frame["error"]["message"], "bad input");
    assert_eq!(frame["error"]["type"], "BadRequestError");
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn completions_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (json!({"model": "other", "prompt": "hi"}), "unknown model"),
        (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
        (
            json!({"model": "model", "prompt": "hi", "max_tokens": 0}),
            "max_tokens=0",
        ),
        (json!({"model": "model", "prompt": ""}), "empty prompt"),
        (
            json!({"model": "model", "prompt": "hi", "best_of": 2}),
            "best_of>1",
        ),
        (
            json!({"model": "model", "prompt": "hi", "suffix": "x"}),
            "suffix",
        ),
        (
            json!({"model": "model", "prompt": "hi", "prompt_embeds": [[1.0]]}),
            "prompt_embeds",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // Malformed JSON → 400 (JsonRejection path).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let response = oneshot(app.clone(), req).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // A closed tm inbox (shutdown) surfaces as 503.
    let app = routes().with_state(app_state(senders_closed()));
    let response = post_json(
        app.clone(),
        "/v1/completions",
        json!({"model": "model", "prompt": "hi"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn chat_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (
            json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}),
            "unknown model",
        ),
        (json!({"model": "model", "messages": []}), "empty messages"),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 0}),
            "n=0",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]}),
            "media content",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "function_call": "auto"}),
            "deprecated function_call",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "audio": {"input_audio": {"data": "x", "format": "wav"}}}),
            "audio",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
            "max_completion_tokens=0",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/chat/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // A valid request with no loaded chat template → 400 (template gate).
    let response = post_json(
        app.clone(),
        "/v1/chat/completions",
        json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn basic_openai_router_excludes_responses_api() {
    let app = routes().with_state(app_state(senders()));
    let response = post_json(app, "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// A closed tm inbox with a *streaming* request must answer inside the
/// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
/// same `error_response` rule the native API applies), not a unary 503.
#[tokio::test]
async fn streaming_submit_failure_answers_inside_the_stream() {
    let app = routes().with_state(app_state(senders_closed()));
    let response = post_json(
        app,
        "/v1/completions",
        json!({"model": "model", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
    assert_eq!(frame["error"]["message"], "service unavailable");
    assert_eq!(frame["error"]["type"], "InternalServerError");
    assert_eq!(frame["error"]["code"], 503);
    assert!(text.contains("[DONE]"));
}
