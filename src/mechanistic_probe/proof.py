from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParsedProof:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


class _Node:
    def __init__(self, head: str) -> None:
        self.head = head


def parse_proof(proof: str) -> ParsedProof:
    """Parse ProofWriter's bracketed proof notation (ported from the paper code)."""
    stack: list[tuple[object, int]] = []
    last_open = 0
    last_open_index = 0
    pending: list[tuple[int, _Node]] = []
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    join_next = False

    for index, token in enumerate(proof.replace("(", " ( ").replace(")", " ) ").split()):
        if token == "(":
            stack.append((token, index))
            last_open, last_open_index = len(stack) - 1, index
        elif token == ")":
            for item, item_index in stack[last_open + 1 :]:
                if isinstance(item, _Node):
                    pending.append((item_index, item))
            stack = stack[:last_open]
            for stack_index, (item, item_index) in enumerate(stack):
                if item == "(":
                    last_open, last_open_index = stack_index, item_index
        elif token in {"[", "]"}:
            continue
        elif token == "->":
            join_next = True
        else:
            if token not in nodes:
                nodes.append(token)
            if join_next:
                retained: list[tuple[int, _Node]] = []
                for item_index, item in pending:
                    if item_index < last_open_index:
                        retained.append((item_index, item))
                    else:
                        edges.append((item.head, token))
                pending = retained
            stack.append((_Node(token), index))
            join_next = False

    # Remove duplicate and immediately reversed edges exactly once.
    cleaned: list[tuple[str, str]] = []
    for edge in edges:
        if edge not in cleaned and (edge[1], edge[0]) not in cleaned:
            cleaned.append(edge)
    return ParsedProof(tuple(nodes), tuple(cleaned))


def labels_for_statements(
    statement_keys: Iterable[str], proof: str, proof_depth: int
) -> tuple[list[int], list[int | None]]:
    """Return useful-membership and proof-height labels in statement order.

    Height is only defined for useful statements.  The convention matches the
    paper's depth-1 label parser: a fact feeding a rule has height 0 and the
    rule has height 1.
    """
    keys = list(statement_keys)
    parsed = parse_proof(proof)
    useful_nodes = [node for node in parsed.nodes if node in keys]
    useful = [int(key in useful_nodes) for key in keys]
    heights: dict[str, int] = {node: 0 for node in useful_nodes}
    if proof_depth > 0:
        for source, target in parsed.edges[:proof_depth]:
            if source in heights and target in heights:
                heights[target] = max(heights[target], heights[source] + 1)
    return useful, [heights.get(key) if key in useful_nodes else None for key in keys]
