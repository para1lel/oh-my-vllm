use std::collections::HashMap;

use super::{CacheDomain, CheckpointId, PageId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub index: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PrefixPath {
    pub checkpoints: Vec<(usize, NodeId, CheckpointId)>,
    pub pages: Vec<(usize, NodeId, PageId)>,
}

struct Node {
    generation: u64,
    edge: Vec<u32>,
    end: usize,
    parent: Option<NodeId>,
    children: HashMap<u32, NodeId>,
    checkpoint: Option<CheckpointId>,
    page: Option<PageId>,
}

#[derive(Default)]
pub struct TokenRadix {
    roots: HashMap<CacheDomain, NodeId>,
    nodes: Vec<Node>,
    free: Vec<u32>,
}

impl TokenRadix {
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes
            .get(id.index as usize)
            .filter(|node| node.generation == id.generation)
    }

    fn node_mut(&mut self, id: NodeId) -> &mut Node {
        let node = &mut self.nodes[id.index as usize];
        assert_eq!(node.generation, id.generation, "stale radix node");
        node
    }

    fn insert(&mut self, edge: Vec<u32>, end: usize, parent: Option<NodeId>) -> NodeId {
        let index = self.free.pop().unwrap_or(self.nodes.len() as u32);
        let generation = self
            .nodes
            .get(index as usize)
            .map_or(1, |node| node.generation + 1);
        let node = Node {
            generation,
            edge,
            end,
            parent,
            children: HashMap::new(),
            checkpoint: None,
            page: None,
        };
        if index as usize == self.nodes.len() {
            self.nodes.push(node);
        } else {
            self.nodes[index as usize] = node;
        }
        NodeId { index, generation }
    }

    pub fn lookup(&self, domain: &CacheDomain, tokens: &[u32]) -> PrefixPath {
        let mut result = PrefixPath::default();
        let Some(&root) = self.roots.get(domain) else {
            return result;
        };
        let mut current = root;
        loop {
            let node = self.node(current).expect("live radix path");
            if let Some(id) = node.checkpoint {
                result.checkpoints.push((node.end, current, id));
            }
            if let Some(id) = node.page {
                result.pages.push((node.end, current, id));
            }
            let Some(token) = tokens.get(node.end) else {
                break;
            };
            let Some(&next) = node.children.get(token) else {
                break;
            };
            let child = self.node(next).expect("live child");
            if tokens.get(node.end..child.end) != Some(child.edge.as_slice()) {
                break;
            }
            current = next;
        }
        result
    }

    pub fn ensure(&mut self, domain: &CacheDomain, tokens: &[u32]) -> NodeId {
        let mut current = if let Some(&root) = self.roots.get(domain) {
            root
        } else {
            let root = self.insert(Vec::new(), 0, None);
            self.roots.insert(domain.clone(), root);
            root
        };
        loop {
            let end = self.node(current).unwrap().end;
            if end == tokens.len() {
                return current;
            }
            let next = self
                .node(current)
                .unwrap()
                .children
                .get(&tokens[end])
                .copied();
            let Some(next) = next else {
                let child = self.insert(tokens[end..].to_vec(), tokens.len(), Some(current));
                self.node_mut(current).children.insert(tokens[end], child);
                return child;
            };
            let edge = &self.node(next).unwrap().edge;
            let common = edge
                .iter()
                .zip(&tokens[end..])
                .take_while(|(a, b)| a == b)
                .count();
            if common == edge.len() {
                current = next;
                continue;
            }
            let split = self.insert(
                tokens[end..end + common].to_vec(),
                end + common,
                Some(current),
            );
            self.node_mut(current).children.insert(tokens[end], split);
            let child = self.node_mut(next);
            child.edge.drain(..common);
            child.parent = Some(split);
            let first = child.edge[0];
            self.node_mut(split).children.insert(first, next);
            current = split;
        }
    }

    pub fn attach_checkpoint(&mut self, node: NodeId, id: CheckpointId) -> Option<CheckpointId> {
        self.node_mut(node).checkpoint.replace(id)
    }

    pub fn attach_page(&mut self, node: NodeId, id: PageId) -> Option<PageId> {
        self.node_mut(node).page.replace(id)
    }

    pub fn detach_checkpoint(&mut self, node: NodeId, id: CheckpointId) {
        if self
            .node(node)
            .is_some_and(|entry| entry.checkpoint == Some(id))
        {
            self.node_mut(node).checkpoint = None;
            self.prune(node);
        }
    }

    pub fn detach_page(&mut self, node: NodeId, id: PageId) {
        if self.node(node).is_some_and(|entry| entry.page == Some(id)) {
            self.node_mut(node).page = None;
            self.prune(node);
        }
    }

    pub fn contains_node(&self, domain: &CacheDomain, mut node: NodeId) -> bool {
        while let Some(entry) = self.node(node) {
            if let Some(parent) = entry.parent {
                node = parent;
            } else {
                return self.roots.get(domain) == Some(&node);
            }
        }
        false
    }

    pub fn end(&self, node: NodeId) -> Option<usize> {
        self.node(node).map(|node| node.end)
    }

    pub fn checkpoint(&self, node: NodeId) -> Option<CheckpointId> {
        self.node(node)?.checkpoint
    }

    fn prune(&mut self, mut id: NodeId) {
        while let Some(node) = self.node(id) {
            if node.checkpoint.is_some() || node.page.is_some() || !node.children.is_empty() {
                break;
            }
            let Some(parent) = node.parent else {
                self.roots.retain(|_, root| *root != id);
                self.node_mut(id).generation += 1;
                self.free.push(id.index);
                break;
            };
            let first = node.edge[0];
            self.node_mut(parent).children.remove(&first);
            self.node_mut(id).generation += 1;
            self.free.push(id.index);
            id = parent;
        }
    }
}
