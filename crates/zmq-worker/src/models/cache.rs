use oh_my_vllm_kv_cache::HybridCoordinator;
use oh_my_vllm_scheduler::policy::CachePolicy;

use crate::client::WorkerClient;

pub trait ServiceCache:
    CachePolicy<
        State = <HybridCoordinator as CachePolicy>::State,
        TransferPlan = <HybridCoordinator as CachePolicy>::TransferPlan,
        Resources = <HybridCoordinator as CachePolicy>::Resources,
        Hit: Send,
        Snapshot: Send,
        Transfer: Send,
    > + Send
    + 'static
{
    fn attach_client(&self, _client: &mut WorkerClient) {}
}

impl ServiceCache for HybridCoordinator {}

impl ServiceCache for super::deepseek_v41::cache::Dep4Cache {
    fn attach_client(&self, client: &mut WorkerClient) {
        client.attach_request_owners(self.owners.clone());
    }
}
