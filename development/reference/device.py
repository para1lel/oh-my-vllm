from oh_my_vllm.worker.device import FUNCTIONS
from oh_my_vllm.worker.device import Device as RuntimeDevice

from development.native_build import build


class Device(RuntimeDevice):
    def __init__(self):
        build("device", FUNCTIONS)
        super().__init__()
