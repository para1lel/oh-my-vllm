from __future__ import annotations

import weakref
from dataclasses import dataclass

from torch import nn


@dataclass
class _RoutedState:
    weights: object = None
    pointers: object = None
    registered: tuple = ()
    generation: int = 0

    def clear_weights(self):
        self.generation += 1
        self.weights = None
        self.pointers = None


def _invalidate_routed_owners(value):
    for owner in getattr(value, "_routed_owners", ()):
        state = owner()
        if state is not None:
            state.clear_weights()


def _unregister_routed_state(state):
    for registration_ref in state.registered:
        registration = registration_ref()
        if registration is None:
            continue
        registration._routed_owners = tuple(
            owner
            for owner in getattr(registration, "_routed_owners", ())
            if owner() is not None and owner() is not state
        )
    state.registered = ()


def _register_routed_owners(state, targets):
    owner_ref = weakref.ref(state)
    registrations = []
    registered_ids = set()
    for target in targets:
        if target is None or id(target) in registered_ids:
            continue
        try:
            registration_ref = weakref.ref(target)
        except TypeError:
            continue
        registered_ids.add(id(target))
        owners = tuple(
            owner
            for owner in getattr(target, "_routed_owners", ())
            if owner() is not None
        )
        if not any(owner() is state for owner in owners):
            target._routed_owners = (*owners, owner_ref)
        registrations.append(registration_ref)
    return tuple(registrations)


class _RoutedModuleList(nn.ModuleList):
    def __setitem__(self, index, module):
        _invalidate_routed_owners(self)
        super().__setitem__(index, module)
