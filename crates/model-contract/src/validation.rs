use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CACHE_SCHEMA_VERSION, CacheGroup, CacheSchema, CacheSource, CacheStorage, ContractError,
    MAX_CACHE_GROUPS, MAX_IDENTIFIER_BYTES, MAX_LAYER_REFERENCES, MODEL_DESCRIPTOR_VERSION,
    ModelDescriptor, ParallelPlan, ParallelStrategy, RequestPlacement, SpeculationSpec,
};

impl ModelDescriptor {
    pub fn validate(&self) -> Result<(), ContractError> {
        require(
            self.schema_version == MODEL_DESCRIPTOR_VERSION,
            "schema_version",
            "unsupported descriptor version",
        )?;
        identifier(&self.architecture, "architecture")?;
        identifier(&self.model_type, "model_type")?;
        for (field, digest) in [
            ("checkpoint_sha256", &self.checkpoint_sha256),
            ("config_sha256", &self.config_sha256),
            ("tokenizer_sha256", &self.tokenizer_sha256),
        ] {
            require(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                field,
                "expected 64 lowercase hexadecimal digits",
            )?;
        }
        require(
            self.vocabulary_size > 0,
            "vocabulary_size",
            "must be positive",
        )?;
        require(
            self.max_context_tokens > 0,
            "max_context_tokens",
            "must be positive",
        )?;
        bounded_unique(&self.capabilities, 8, "capabilities")?;
        bounded_unique(&self.weight_formats, 5, "weight_formats")?;
        require(
            !self.weight_formats.is_empty(),
            "weight_formats",
            "must not be empty",
        )?;
        require(
            self.schema_version == self.cache_schema.schema_version,
            "cache_schema.schema_version",
            "must match the descriptor version",
        )?;
        self.cache_schema.validate()?;
        self.parallel_plan.validate()?;
        require(
            self.speculation.len() <= 2,
            "speculation",
            "too many methods",
        )?;
        let methods: Vec<_> = self.speculation.iter().map(|spec| spec.method).collect();
        bounded_unique(&methods, 2, "speculation")?;
        for spec in &self.speculation {
            spec.validate()?;
        }
        Ok(())
    }
}

impl CacheSchema {
    pub fn validate(&self) -> Result<(), ContractError> {
        require(
            self.schema_version == CACHE_SCHEMA_VERSION,
            "cache_schema.schema_version",
            "unsupported cache version",
        )?;
        require(
            !self.groups.is_empty() && self.groups.len() <= MAX_CACHE_GROUPS,
            "cache_schema.groups",
            "expected 1 to 256 groups",
        )?;
        let groups: BTreeMap<_, _> = self
            .groups
            .iter()
            .map(|group| (group.name.as_str(), group))
            .collect();
        require(
            groups.len() == self.groups.len(),
            "cache_schema.groups",
            "duplicate name",
        )?;
        let mut layer_count = 0;
        for group in &self.groups {
            group.validate()?;
            layer_count += group.layers.len();
            require(
                layer_count <= MAX_LAYER_REFERENCES,
                "cache_schema.groups.layers",
                "too many layer references",
            )?;
            validate_source(group, &groups)?;
        }
        Ok(())
    }
}

impl CacheGroup {
    fn validate(&self) -> Result<(), ContractError> {
        identifier(&self.name, "cache_group.name")?;
        require(
            self.block_tokens > 0,
            "cache_group.block_tokens",
            "must be positive",
        )?;
        require(
            self.capacity_blocks > 0,
            "cache_group.capacity_blocks",
            "must be positive",
        )?;
        require(
            !self.layers.is_empty(),
            "cache_group.layers",
            "must not be empty",
        )?;
        bounded_unique(&self.layers, MAX_LAYER_REFERENCES, "cache_group.layers")?;
        if let CacheStorage::Layout { id, version } = &self.storage {
            identifier(id, "cache_group.storage.id")?;
            require(
                *version > 0,
                "cache_group.storage.version",
                "must be positive",
            )?;
        }
        if let CacheSource::Shared { producer } = &self.source {
            identifier(producer, "cache_group.source.producer")?;
        }
        Ok(())
    }
}

impl ParallelPlan {
    pub fn validate(&self) -> Result<(), ContractError> {
        let expected_size = match self.strategy {
            ParallelStrategy::Tp1 => 1,
            ParallelStrategy::Tp4 | ParallelStrategy::Tep4 | ParallelStrategy::Dep4 => 4,
        };
        require(
            self.world_size == expected_size,
            "parallel_plan.world_size",
            "must match the selected strategy",
        )?;
        bounded_unique(&self.ranks, 4, "parallel_plan.ranks")?;
        require(
            self.ranks.len() == expected_size as usize
                && self.ranks.iter().all(|&rank| rank < expected_size),
            "parallel_plan.ranks",
            "expected each logical rank from zero to world_size minus one",
        )?;
        let placement = match self.strategy {
            ParallelStrategy::Dep4 => RequestPlacement::RankPartitioned,
            _ => RequestPlacement::Replicated,
        };
        require(
            self.request_placement == placement,
            "parallel_plan.request_placement",
            "must match the selected strategy",
        )
    }
}

impl SpeculationSpec {
    pub fn validate(&self) -> Result<(), ContractError> {
        require(
            self.predictor_stages > 0,
            "speculation.predictor_stages",
            "must be positive",
        )?;
        require(
            self.max_draft_tokens > 0,
            "speculation.max_draft_tokens",
            "must be positive",
        )?;
        require(
            self.max_draft_tokens.checked_add(1) == Some(self.max_verification_rows),
            "speculation.max_verification_rows",
            "must equal max_draft_tokens plus one without overflow",
        )
    }
}

fn validate_source<'a>(
    mut group: &'a CacheGroup,
    groups: &BTreeMap<&str, &'a CacheGroup>,
) -> Result<(), ContractError> {
    for _ in 0..groups.len() {
        match &group.source {
            CacheSource::Owned {} => return Ok(()),
            CacheSource::Shared { producer } => {
                group = groups
                    .get(producer.as_str())
                    .copied()
                    .ok_or(ContractError {
                        field: "cache_group.source.producer",
                        reason: "unknown producer",
                    })?;
            }
        }
    }
    Err(ContractError {
        field: "cache_group.source.producer",
        reason: "cyclic producer reference",
    })
}

fn identifier(value: &str, field: &'static str) -> Result<(), ContractError> {
    require(
        !value.is_empty()
            && value.len() <= MAX_IDENTIFIER_BYTES
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
        field,
        "expected 1 to 128 ASCII letters, digits, underscores, dots, or hyphens",
    )
}

fn bounded_unique<T: Ord>(
    values: &[T],
    limit: usize,
    field: &'static str,
) -> Result<(), ContractError> {
    require(values.len() <= limit, field, "too many entries")?;
    require(
        values.iter().collect::<BTreeSet<_>>().len() == values.len(),
        field,
        "duplicate entry",
    )
}

fn require(valid: bool, field: &'static str, reason: &'static str) -> Result<(), ContractError> {
    if valid {
        Ok(())
    } else {
        Err(ContractError { field, reason })
    }
}
