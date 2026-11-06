use super::*;

#[test]
fn long_resident_prefills_keep_real_token_intervals() {
    for budget in [3, 784, 32768] {
        let mut scheduler = production_long_scheduler(4);
        scheduler.config.max_num_batched_tokens = budget;
        for (id, length) in [(1, 317), (2, 1234), (3, 1569)] {
            assert!(scheduler.add_request(make_req(id, length, 2)));
        }
        finish_prefills(&mut scheduler);
        assert_eq!(scheduler.num_running() + scheduler.num_waiting(), 0);
        assert_eq!(scheduler.cache_resource_counts().1, 1399);
    }
}

#[test]
fn resident_prefixes_isolate_namespace_model_and_media_salt() {
    use oh_my_vllm_kv_cache::hash::BlockHash;
    let mut scheduler = make_scheduler(64, 16);
    let mut source = make_req(1, 9, 1);
    source.cache_domain.namespace = "a".into();
    source.cache_salt = Some(BlockHash(11));
    assert!(scheduler.add_request(source));
    finish_prefills(&mut scheduler);
    for (namespace, model, salt, expected) in [
        ("a", "", 11, 8),
        ("b", "", 11, 0),
        ("a", "", 12, 0),
        ("a", "other", 11, 0),
    ] {
        let mut request = make_req(2, 9, 1);
        request.cache_domain.namespace = namespace.into();
        request.cache_domain.model = model.into();
        request.cache_salt = Some(BlockHash(salt));
        scheduler.kv.initialize_hashes(&mut request);
        assert_eq!(scheduler.kv.find_prefix(&request).length, expected);
    }
}

#[test]
fn partial_resident_checkpoint_reuses_prefix_without_host_state() {
    let mut scheduler = production_long_scheduler(2);
    assert!(scheduler.add_request(make_req(1, 317, 1)));
    finish_prefills(&mut scheduler);
    assert!(scheduler.add_request(make_req(2, 317, 1)));
    let batch = scheduler.schedule();
    assert_eq!(batch.cache_hit_tokens, 316);
    assert_eq!(batch.scheduled[0].token_ids, [316]);
    assert_eq!(batch.scheduled[0].work.sampling, SamplingMode::Target);
}
