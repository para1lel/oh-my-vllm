"""Validation for the current paired-cache transfer wire messages."""


def identifier(value, name):
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"invalid cache {name}")
    return value


def validate_operation(operation):
    for name in ("transfer_id", "snapshot_id"):
        identifier(operation[name], name)
    if operation["request_id"] is not None:
        identifier(operation["request_id"], "request_id")
    if operation["kind"] not in ("store", "load", "drop"):
        raise ValueError("invalid cache operation")
    for name in ("fa_blocks", "mamba_blocks"):
        blocks = operation[name]
        if not isinstance(blocks, list) or any(
            type(block) is not int or not 0 < block <= 0xFFFFFFFF for block in blocks
        ):
            raise ValueError("invalid cache block list")
        if len(blocks) != len(set(blocks)):
            raise ValueError("cache blocks must be unique")
    if operation["kind"] == "drop" and (
        operation["fa_blocks"] or operation["mamba_blocks"]
    ):
        raise ValueError("cache drop cannot contain blocks")


def completion(transfer_id, status, size=0):
    return {"transfer_id": transfer_id, "status": status, "bytes": size}
