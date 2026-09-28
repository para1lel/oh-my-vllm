//! Error types for the ZMQ worker crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("ZMQ error: {0}")]
    Zmq(#[from] zeromq::ZmqError),

    #[error("msgpack encode error: {0}")]
    MsgpackEncode(#[from] rmp_serde::encode::Error),

    #[error("msgpack decode error: {0}")]
    MsgpackDecode(#[from] rmp_serde::decode::Error),

    #[error("Python worker returned an error: {0}")]
    WorkerError(String),

    #[error("invalid request rejected by Python worker: {0}")]
    WorkerValidation(String),

    #[error("unexpected message type '{0}' from Python worker")]
    UnexpectedMessageType(String),

    #[error("Python worker process exited unexpectedly")]
    WorkerDied,

    #[error("timeout waiting for Python worker")]
    Timeout,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
