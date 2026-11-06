use super::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotKey {
    pub epoch: u64,
    pub slot: u32,
    pub version: u64,
}

#[derive(Clone)]
struct Slot {
    version: u64,
    pins: usize,
    attachment: Option<NodeId>,
    touched: u64,
}

#[derive(Clone)]
pub struct SlotPool {
    epoch: u64,
    first: u32,
    clock: u64,
    slots: Vec<Slot>,
}

pub struct PoolReservation {
    keys: Vec<SlotKey>,
    pub detached: Vec<(NodeId, SlotKey)>,
}

impl PoolReservation {
    pub fn keys(&self) -> &[SlotKey] {
        &self.keys
    }
}

impl SlotPool {
    pub fn new(epoch: u64, first: u32, capacity: u32) -> Self {
        assert!(first > 0);
        Self {
            epoch,
            first,
            clock: 0,
            slots: vec![
                Slot {
                    version: 0,
                    pins: 0,
                    attachment: None,
                    touched: 0
                };
                capacity as usize
            ],
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn available(&self) -> usize {
        self.slots.iter().filter(|slot| slot.pins == 0).count()
    }

    pub fn cached(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.attachment.is_some())
            .count()
    }

    fn index(&self, key: SlotKey) -> Option<usize> {
        let index = key.slot.checked_sub(self.first)? as usize;
        let slot = self.slots.get(index)?;
        (key.epoch == self.epoch && slot.version == key.version && key.version > 0).then_some(index)
    }

    pub fn contains(&self, key: SlotKey) -> bool {
        self.index(key).is_some()
    }

    pub fn pin(&mut self, key: SlotKey) -> bool {
        let Some(index) = self.index(key) else {
            return false;
        };
        self.slots[index].pins += 1;
        true
    }

    pub fn unpin(&mut self, key: SlotKey) {
        let index = self.index(key).expect("stale slot release");
        self.slots[index].pins = self.slots[index]
            .pins
            .checked_sub(1)
            .expect("unbalanced pin");
    }

    pub fn pins(&self, key: SlotKey) -> usize {
        self.index(key).map_or(0, |index| self.slots[index].pins)
    }

    pub fn reserve(&mut self, count: usize) -> Option<PoolReservation> {
        if count > self.available() {
            return None;
        }
        let mut candidates = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.pins == 0)
            .map(|(index, slot)| (slot.attachment.is_some(), slot.touched, index))
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        let mut result = PoolReservation {
            keys: Vec::new(),
            detached: Vec::new(),
        };
        for (_, _, index) in candidates.into_iter().take(count) {
            self.clock = self.clock.checked_add(1).expect("slot clock exhausted");
            let slot = &mut self.slots[index];
            let old = SlotKey {
                epoch: self.epoch,
                slot: self.first + index as u32,
                version: slot.version,
            };
            if let Some(node) = slot.attachment.take() {
                result.detached.push((node, old));
            }
            slot.version = slot.version.checked_add(1).expect("slot version exhausted");
            slot.pins = 1;
            slot.touched = self.clock;
            result.keys.push(SlotKey {
                version: slot.version,
                ..old
            });
        }
        Some(result)
    }

    pub fn attach(&mut self, key: SlotKey, node: NodeId) {
        let index = self.index(key).expect("stale attachment");
        assert!(self.slots[index].attachment.is_none_or(|old| old == node));
        self.slots[index].attachment = Some(node);
    }

    pub fn detach(&mut self, key: SlotKey) -> Option<NodeId> {
        let index = self.index(key)?;
        self.slots[index].attachment.take()
    }
}
