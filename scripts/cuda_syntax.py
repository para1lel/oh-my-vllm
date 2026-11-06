"""Adapt known CUDA annotations for the pinned concrete-syntax grammar.

These changes apply only to a parser copy. Every byte offset and newline stays
in place. Tree-sitter still finds functions and rejects remaining syntax errors.
"""

OPAQUE = {
    "comment",
    "string_literal",
    "char_literal",
    "raw_string_literal",
    "system_lib_string",
    "preproc_arg",
}
QUALIFIERS = {
    b"CUTE_HOST_DEVICE": b"inline",
    b"CUTE_DEVICE": b"inline",
    b"CUTE_HOST": b"inline",
    b"CUTLASS_HOST_DEVICE": b"inline",
    b"CUTLASS_DEVICE": b"inline",
    b"CUDART_CB": b"__cdecl",
    b"TOPK_SELECT_GM": b"",
    b"CUTE_UNROLL": b"",
    b"CUTLASS_PRAGMA_UNROLL": b"",
}
ALIGNMENT = {b"__align__", b"CUTE_ALIGNAS"}
OBJECT_ANNOTATIONS = {
    b"TVM_FFI_DECLARE_OBJECT_INFO_FINAL",
    b"TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE",
}


def nodes(root, opaque=OPAQUE):
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        if node.type not in opaque:
            pending.extend(reversed(node.children))


def tokens(root):
    opaque = OPAQUE | {"preproc_def", "preproc_function_def"}
    return [
        node
        for node in nodes(root, opaque)
        if not node.children and node.type not in opaque and not node.is_missing
    ]


def blank(data: bytearray, start: int, end: int, prefix: bytes = b"") -> None:
    data[start:end] = bytes(
        byte if byte in (10, 13) else 32 for byte in data[start:end]
    )
    if (
        b"\n" in data[start : start + len(prefix)]
        or b"\r" in data[start : start + len(prefix)]
    ):
        raise ValueError("annotation replacement crosses a physical line")
    data[start : start + len(prefix)] = prefix


def closing_parenthesis(entries: list, index: int) -> int | None:
    if index + 1 == len(entries) or entries[index + 1].text != b"(":
        return None
    depth = 0
    for end in range(index + 1, len(entries)):
        text = entries[end].text
        depth += int(text == b"(") - int(text == b")")
        if depth == 0:
            return end
    return None


def annotation_arguments(
    entries: list, index: int, end: int, source: bytes, parser
) -> bool:
    if any(node.text in {b"{", b"}"} for node in entries[index + 1 : end]):
        return False
    arguments = source[entries[index + 1].start_byte : entries[end].end_byte]
    syntax = parser.parse(b"void __size_annotation__(){call" + arguments + b";}")
    if syntax.root_node.has_error:
        return False
    expected = {1, 2, 3} if entries[index].text == b"__launch_bounds__" else {3}
    calls = [node for node in nodes(syntax.root_node) if node.type == "call_expression"]
    return bool(
        calls
        and calls[0].child_by_field_name("arguments").named_child_count in expected
    )


def normalize_annotations(source: bytes, parser) -> bytes:
    data = bytearray(source)
    syntax = parser.parse(source)
    for node in nodes(syntax.root_node):
        if node.type == "comment" and not node.has_error:
            blank(data, node.start_byte, node.end_byte)
    syntax = parser.parse(data)
    for node in nodes(syntax.root_node):
        directive = node.child_by_field_name("directive")
        if node.type == "preproc_call" and directive and directive.text == b"#pragma":
            blank(data, node.start_byte, node.end_byte)
    syntax = parser.parse(data)
    entries = tokens(syntax.root_node)
    for index, node in enumerate(entries):
        text = node.text
        if text in QUALIFIERS:
            blank(data, node.start_byte, node.end_byte, QUALIFIERS[text])
        elif text in ALIGNMENT:
            blank(data, node.start_byte, node.end_byte, b"alignas")
        elif text in OBJECT_ANNOTATIONS or text == b"__launch_bounds__":
            end = closing_parenthesis(entries, index)
            if end is not None and annotation_arguments(
                entries, index, end, bytes(data), parser
            ):
                prefix = b"[[maybe_unused]]" if text == b"__launch_bounds__" else b""
                blank(data, node.start_byte, entries[end].end_byte, prefix)
    return bytes(data)


def misplaced_dependent_type(node) -> bool:
    if node.type != "dependent_type":
        return False
    ancestor = node.parent
    while ancestor and ancestor.type in {"parameter_declaration", "parameter_list"}:
        ancestor = ancestor.parent
    return bool(
        ancestor
        and ancestor.type == "function_declarator"
        and ancestor.has_error
        and ancestor.parent.type == "function_definition"
        and ancestor.parent.parent.type == "compound_statement"
    )


def normalize_dependent_constructions(source: bytes, parser) -> bytes:
    syntax = parser.parse(source)
    errors = {
        node.start_byte
        for node in nodes(syntax.root_node)
        if (node.is_error and node.text == b"typename")
        or misplaced_dependent_type(node)
    }
    if not errors:
        return source
    data = bytearray(source)
    entries = tokens(syntax.root_node)
    for index, node in enumerate(entries):
        if node.start_byte not in errors or index == 0:
            continue
        if entries[index - 1].text not in {b"(", b",", b"=", b"return"}:
            continue
        end = index + 1
        qualified = False
        while end + 2 < len(entries) and entries[end].type in {
            "identifier",
            "type_identifier",
            "namespace_identifier",
        }:
            if entries[end + 1].text == b"::":
                qualified = True
                end += 2
            else:
                end += 1
                break
        if (
            qualified
            and end + 1 < len(entries)
            and entries[end].text == b"{"
            and entries[end + 1].text == b"}"
        ):
            blank(data, node.start_byte, node.end_byte)
    return bytes(data)


def parser_source(source: bytes, parser) -> bytes:
    return normalize_dependent_constructions(
        normalize_annotations(source, parser), parser
    )
