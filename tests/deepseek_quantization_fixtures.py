import torch


class DeepSeekQuantizationFixture:
    def setUp(self):
        device = torch.device(self.device)
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)
