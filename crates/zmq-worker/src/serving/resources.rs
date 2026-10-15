use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub const HTTP_BODY_BYTES: usize = 8 << 20;
const QUEUED_LIMIT: usize = 64;
const LIVE_LIMIT: usize = 64;

#[derive(Clone)]
pub struct Budget(Arc<BudgetInner>);

struct BudgetInner {
    queued: Arc<Semaphore>,
    live: Arc<Semaphore>,
    state: Mutex<BudgetState>,
    max_bytes: usize,
}

#[derive(Default)]
struct BudgetState {
    bytes: usize,
    drains: usize,
}

#[derive(Clone)]
pub struct Lease(Arc<LeaseInner>);

struct LeaseInner {
    budget: Budget,
    state: Mutex<LeaseState>,
    pub received: Instant,
    pub received_unix_s: f64,
    pub deadline: Instant,
    pub epoch: u64,
}

struct LeaseState {
    bytes: usize,
    queued: Option<OwnedSemaphorePermit>,
    live: Option<OwnedSemaphorePermit>,
}

impl Budget {
    pub fn new(max_bytes: usize) -> Self {
        Self(Arc::new(BudgetInner {
            queued: Arc::new(Semaphore::new(QUEUED_LIMIT)),
            live: Arc::new(Semaphore::new(LIVE_LIMIT)),
            state: Mutex::new(BudgetState::default()),
            max_bytes,
        }))
    }

    pub fn admit(&self, timeout: Duration, epoch: u64) -> Result<Lease> {
        let queued = self.0.queued.clone().try_acquire_owned()?;
        let received = Instant::now();
        let lease = Lease(Arc::new(LeaseInner {
            budget: self.clone(),
            state: Mutex::new(LeaseState {
                bytes: 0,
                queued: Some(queued),
                live: None,
            }),
            received,
            received_unix_s: unix_seconds(),
            deadline: received + timeout,
            epoch,
        }));
        lease.resize(HTTP_BODY_BYTES)?;
        Ok(lease)
    }

    pub fn snapshot(&self) -> Value {
        let state = self.0.state.lock().unwrap();
        json!({
            "request_state_bytes": state.bytes,
            "request_state_max_bytes": self.0.max_bytes,
            "queued_reservations": QUEUED_LIMIT - self.0.queued.available_permits(),
            "live_reservations": LIVE_LIMIT - self.0.live.available_permits(),
            "completed_stream_drains": state.drains,
        })
    }
}

impl Lease {
    pub fn received(&self) -> Instant {
        self.0.received
    }

    pub fn received_unix_s(&self) -> f64 {
        self.0.received_unix_s
    }

    pub fn deadline(&self) -> Instant {
        self.0.deadline
    }

    pub fn epoch(&self) -> u64 {
        self.0.epoch
    }

    pub fn resize(&self, bytes: usize) -> Result<()> {
        let mut lease = self.0.state.lock().unwrap();
        let mut budget = self.0.budget.0.state.lock().unwrap();
        let total = budget
            .bytes
            .saturating_sub(lease.bytes)
            .saturating_add(bytes);
        ensure!(
            total <= self.0.budget.0.max_bytes,
            "request state byte budget exhausted"
        );
        budget.bytes = total;
        lease.bytes = bytes;
        Ok(())
    }

    pub fn grow(&self, additional: usize) -> Result<()> {
        let mut lease = self.0.state.lock().unwrap();
        let mut budget = self.0.budget.0.state.lock().unwrap();
        let bytes = budget.bytes.checked_add(additional);
        ensure!(
            bytes.is_some_and(|value| value <= self.0.budget.0.max_bytes),
            "request state byte budget exhausted"
        );
        lease.bytes = lease
            .bytes
            .checked_add(additional)
            .ok_or_else(|| anyhow::anyhow!("request state byte reservation overflow"))?;
        budget.bytes = bytes.unwrap();
        Ok(())
    }

    pub fn start(&self) -> Result<()> {
        let mut lease = self.0.state.lock().unwrap();
        ensure!(
            Instant::now() < self.0.deadline,
            "request deadline exceeded in queue"
        );
        lease.live = Some(self.0.budget.0.live.clone().try_acquire_owned()?);
        lease.queued = None;
        Ok(())
    }

    pub fn drain(&self) -> Drain {
        self.0.budget.0.state.lock().unwrap().drains += 1;
        Drain(self.clone())
    }
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let bytes = self.state.get_mut().unwrap().bytes;
        let mut budget = self.budget.0.state.lock().unwrap();
        budget.bytes -= bytes;
    }
}

pub struct Drain(Lease);

impl Drop for Drain {
    fn drop(&mut self) {
        self.0.0.budget.0.state.lock().unwrap().drains -= 1;
    }
}

pub fn serialized_bytes<T: serde::Serialize>(value: &T) -> Result<usize> {
    struct Counter(usize);

    impl Write for Counter {
        fn write(&mut self, input: &[u8]) -> io::Result<usize> {
            self.0 = self
                .0
                .checked_add(input.len())
                .ok_or_else(|| io::Error::other("serialized request size overflow"))?;
            Ok(input.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value)?;
    Ok(counter.0)
}

pub fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub fn rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        // SAFETY: sysconf only returns the system page-size scalar.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(page_size).ok()?.checked_mul(resident)
    }
    #[cfg(not(target_os = "linux"))]
    None
}
