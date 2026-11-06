import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from oh_my_vllm.operators.output.api import HiddenState, Inputs, Outputs, Weights
from oh_my_vllm.operators.output.workspace import Workspace


class CachedOutput(unittest.TestCase):
    def workspace(self, *, closed=False, hidden_only=False):
        workspace = Workspace.__new__(Workspace)
        workspace.closed = closed
        workspace.plan = SimpleNamespace(hidden_only=hidden_only)
        workspace.module = SimpleNamespace(
            sample_project_hidden=Mock(), final_head_hidden=Mock()
        )
        workspace.outputs = Outputs(object(), object())
        return workspace

    def test_projects_saved_normalized_hidden_without_norm_or_rng(self):
        workspace = self.workspace()
        hidden, projection = object(), object()
        result = workspace.project_hidden(Weights(object(), projection), hidden)
        self.assertIs(result, workspace.outputs)
        workspace.module.sample_project_hidden.assert_called_once_with(
            hidden, projection, workspace.outputs.logits
        )

    def test_hidden_only_workspace_exports_head_ready_cache_without_projection(self):
        workspace = self.workspace(hidden_only=True)
        value, residual, norm, cache, slots = (object() for _ in range(5))
        result = workspace.head_hidden(
            Weights(norm), Inputs(value, residual), HiddenState(cache, slots)
        )
        self.assertIs(result, workspace.outputs)
        workspace.module.final_head_hidden.assert_called_once_with(
            value, residual, norm, workspace.outputs.hidden, cache, slots
        )
        workspace.module.sample_project_hidden.assert_not_called()

    def test_closed_or_hidden_only_workspace_rejects_projection(self):
        for closed, hidden_only, failure in (
            (True, False, RuntimeError),
            (False, True, ValueError),
        ):
            with self.subTest(closed=closed, hidden_only=hidden_only):
                workspace = self.workspace(closed=closed, hidden_only=hidden_only)
                with self.assertRaises(failure):
                    workspace.project_hidden(Weights(object(), object()), object())
                workspace.module.sample_project_hidden.assert_not_called()
