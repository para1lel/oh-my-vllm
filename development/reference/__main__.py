import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .cases import numerical_cases
from .checkpoint import Checkpoint
from .checks import verify_case
from .device import Device
from .providers import prepare


def check_case(device, case, shape):
    results = {}
    for mode in ("eager", "graph"):
        case.reset()
        if mode == "eager":
            result = case.check()
        else:
            with device.graph(case.run) as replay:
                case.reset()
                replay()
                device.synchronize()
                result = case.check(execute=False)
        if shape.rows:
            verify_case(result, shape.operator, rows=shape.rows, variant=shape.variant)
        elif any(
            result.get(key) is not True
            for key in ("passed", "empty_rows", "unchanged_slots")
        ):
            raise ValueError("empty-input guard failed")
        results[mode] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--operator", choices=sorted({shape.operator for shape in numerical_cases()})
    )
    arguments = parser.parse_args()
    device = Device()
    checkpoint = Checkpoint(arguments.model)
    shapes = [
        shape
        for shape in numerical_cases()
        if arguments.operator in (None, shape.operator)
    ]
    try:
        for operator in sorted({shape.operator for shape in shapes}):
            make_case = prepare(operator, device, checkpoint)
            for shape in shapes:
                if shape.operator == operator:
                    case = make_case(shape)
                    try:
                        result = check_case(device, case, shape)
                    finally:
                        close = getattr(case, "close", None)
                        if close is not None:
                            close()
                    checkpoint.verify_identity()
                    print(
                        json.dumps(dict(shape=asdict(shape), checks=result)), flush=True
                    )
                    del case
            del make_case
    finally:
        device.synchronize()


if __name__ == "__main__":
    main()
