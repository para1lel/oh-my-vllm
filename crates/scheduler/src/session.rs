use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionSource {
    Explicit,
    Correlation,
    ResponsesRoot,
    Anonymous,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId {
    pub scope: String,
    pub source: SessionSource,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestIdentity {
    pub session: SessionId,
    pub namespace: String,
}

impl RequestIdentity {
    pub fn anonymous(request: u64) -> Self {
        Self {
            session: SessionId {
                scope: String::new(),
                source: SessionSource::Anonymous,
                value: request.to_string(),
            },
            namespace: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdmissionGeneration(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestTicket {
    pub request: u64,
    pub identity: RequestIdentity,
    pub generation: AdmissionGeneration,
}

#[derive(Default)]
pub(crate) struct SessionRegistry {
    sequence: u64,
    entries: HashMap<SessionId, Entry>,
    pending: HashSet<(u64, u64)>,
}

struct Entry {
    latest: AdmissionGeneration,
    owners: HashSet<u64>,
}

impl SessionRegistry {
    pub fn empty_like(&self) -> Self {
        Self {
            sequence: self.sequence,
            ..Self::default()
        }
    }

    pub fn reserve(&mut self, identity: RequestIdentity, request: u64) -> RequestTicket {
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("admission sequence exhausted");
        let generation = AdmissionGeneration(self.sequence);
        let entry = self
            .entries
            .entry(identity.session.clone())
            .or_insert(Entry {
                latest: generation,
                owners: HashSet::new(),
            });
        entry.latest = generation;
        entry.owners.insert(generation.0);
        self.pending.insert((request, generation.0));
        RequestTicket {
            request,
            identity,
            generation,
        }
    }

    pub fn consume(&mut self, ticket: &RequestTicket) -> bool {
        self.entries
            .get(&ticket.identity.session)
            .is_some_and(|entry| entry.owners.contains(&ticket.generation.0))
            && self.pending.remove(&(ticket.request, ticket.generation.0))
    }

    pub fn is_latest(&self, ticket: &RequestTicket) -> bool {
        self.entries
            .get(&ticket.identity.session)
            .is_some_and(|entry| {
                entry.latest == ticket.generation && entry.owners.contains(&ticket.generation.0)
            })
    }

    pub fn release_pending(&mut self, ticket: &RequestTicket) -> bool {
        if !self
            .pending
            .contains(&(ticket.request, ticket.generation.0))
        {
            return false;
        }
        self.release(ticket)
    }

    pub fn release(&mut self, ticket: &RequestTicket) -> bool {
        self.pending.remove(&(ticket.request, ticket.generation.0));
        let Some(entry) = self.entries.get_mut(&ticket.identity.session) else {
            return false;
        };
        let removed = entry.owners.remove(&ticket.generation.0);
        if entry.owners.is_empty() {
            self.entries.remove(&ticket.identity.session);
        }
        removed
    }
}
