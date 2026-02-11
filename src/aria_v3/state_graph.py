"""
State graph: tracks game states and transitions for deduplication and dead-end pruning.

Nodes are frame hashes. Edges are (action_id, result, target_hash).
The graph prevents the agent from repeating failed actions and enables
navigation back to states with untested actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum


class EdgeResult(IntEnum):
    UNTESTED = 0
    SUCCESS = 1   # frame changed
    DEAD = -1     # self-loop or game-over


@dataclass
class Edge:
    action_id: int          # action index in the agent's action space
    result: EdgeResult = EdgeResult.UNTESTED
    target_hash: str = ""   # frame hash of resulting state


@dataclass
class Node:
    frame_hash: str
    edges: list[Edge] = field(default_factory=list)
    predecessors: list[tuple[str, int]] = field(default_factory=list)  # (from_hash, edge_idx)


class StateGraph:
    """Graph of game states for exploration and dead-end pruning."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.level_start_hash: str | None = None

    def register_state(self, frame_hash: str, num_actions: int) -> None:
        """Register a state if not already known.

        Args:
            frame_hash: Hash of the current frame
            num_actions: Total number of actions available from this state
        """
        if frame_hash in self.nodes:
            return

        node = Node(frame_hash=frame_hash)
        for i in range(num_actions):
            node.edges.append(Edge(action_id=i))
        self.nodes[frame_hash] = node

        # First state registered is the level start
        if self.level_start_hash is None:
            self.level_start_hash = frame_hash

    def update(
        self, from_hash: str, action_id: int, to_hash: str, frame_changed: bool
    ) -> None:
        """Record the result of taking an action.

        Args:
            from_hash: Frame hash before action
            action_id: Which action was taken
            to_hash: Frame hash after action
            frame_changed: Whether the frame changed
        """
        if from_hash not in self.nodes:
            return

        node = self.nodes[from_hash]
        if action_id >= len(node.edges):
            return

        edge = node.edges[action_id]

        if not frame_changed:
            # Self-loop: action does nothing
            edge.result = EdgeResult.DEAD
            edge.target_hash = from_hash
        elif to_hash == self.level_start_hash and from_hash != self.level_start_hash:
            # Returns to level start = likely game-over/reset
            edge.result = EdgeResult.DEAD
            edge.target_hash = to_hash
        else:
            edge.result = EdgeResult.SUCCESS
            edge.target_hash = to_hash
            # Record predecessor for backward propagation
            if to_hash in self.nodes:
                to_node = self.nodes[to_hash]
                to_node.predecessors.append((from_hash, action_id))

        # Check if all edges are dead → propagate backward
        if edge.result == EdgeResult.DEAD:
            self._maybe_propagate_dead(from_hash)

    def _maybe_propagate_dead(self, frame_hash: str) -> None:
        """If all edges from a node are dead, mark incoming edges as dead too."""
        node = self.nodes.get(frame_hash)
        if node is None:
            return

        all_dead = all(e.result == EdgeResult.DEAD for e in node.edges)
        if not all_dead:
            return

        # Propagate backward
        for pred_hash, pred_edge_idx in node.predecessors:
            pred_node = self.nodes.get(pred_hash)
            if pred_node is None:
                continue
            if pred_edge_idx < len(pred_node.edges):
                pred_edge = pred_node.edges[pred_edge_idx]
                if pred_edge.result != EdgeResult.DEAD:
                    pred_edge.result = EdgeResult.DEAD
                    self._maybe_propagate_dead(pred_hash)

    def get_untested_action(self, frame_hash: str) -> int | None:
        """Get an untested action from this state, or None if all tested.

        Returns:
            Action index, or None
        """
        node = self.nodes.get(frame_hash)
        if node is None:
            return None

        for edge in node.edges:
            if edge.result == EdgeResult.UNTESTED:
                return edge.action_id
        return None

    def get_live_action(self, frame_hash: str) -> int | None:
        """Get any non-dead action from this state (for navigation).

        Returns:
            Action index of a SUCCESS edge, or None
        """
        node = self.nodes.get(frame_hash)
        if node is None:
            return None

        for edge in node.edges:
            if edge.result == EdgeResult.SUCCESS:
                return edge.action_id
        return None

    def get_path_to_frontier(self, from_hash: str) -> int | None:
        """BFS to find the shortest path to a state with untested actions.

        Returns the first action to take, or None if no frontier reachable.
        """
        if from_hash not in self.nodes:
            return None

        # Check if current node has untested actions
        untested = self.get_untested_action(from_hash)
        if untested is not None:
            return untested

        # BFS through SUCCESS edges
        visited = {from_hash}
        # Queue entries: (current_hash, first_action_taken)
        queue: deque[tuple[str, int]] = deque()

        # Seed with SUCCESS edges from current node
        node = self.nodes[from_hash]
        for edge in node.edges:
            if edge.result == EdgeResult.SUCCESS and edge.target_hash not in visited:
                visited.add(edge.target_hash)
                queue.append((edge.target_hash, edge.action_id))

        while queue:
            current_hash, first_action = queue.popleft()
            current_node = self.nodes.get(current_hash)
            if current_node is None:
                continue

            # Check if this node has untested actions
            if any(e.result == EdgeResult.UNTESTED for e in current_node.edges):
                return first_action  # Take this first step toward the frontier

            # Expand
            for edge in current_node.edges:
                if edge.result == EdgeResult.SUCCESS and edge.target_hash not in visited:
                    visited.add(edge.target_hash)
                    queue.append((edge.target_hash, first_action))

        return None  # No frontier reachable

    def get_full_path_to_frontier(self, from_hash: str) -> list[int] | None:
        """BFS to find shortest path to a state with untested actions.

        Returns list of action indices forming the full path, or None.
        """
        if from_hash not in self.nodes:
            return None

        # Check if current node has untested actions
        untested = self.get_untested_action(from_hash)
        if untested is not None:
            return [untested]

        # BFS through SUCCESS edges, tracking full path
        visited = {from_hash}
        # Queue entries: (current_hash, path of action indices)
        queue: deque[tuple[str, list[int]]] = deque()

        node = self.nodes[from_hash]
        for edge in node.edges:
            if edge.result == EdgeResult.SUCCESS and edge.target_hash not in visited:
                visited.add(edge.target_hash)
                queue.append((edge.target_hash, [edge.action_id]))

        while queue:
            current_hash, path = queue.popleft()
            current_node = self.nodes.get(current_hash)
            if current_node is None:
                continue

            if any(e.result == EdgeResult.UNTESTED for e in current_node.edges):
                # Also append the untested action at the frontier
                for e in current_node.edges:
                    if e.result == EdgeResult.UNTESTED:
                        return path + [e.action_id]

            # Limit BFS depth to avoid very long paths
            if len(path) > 30:
                continue

            for edge in current_node.edges:
                if edge.result == EdgeResult.SUCCESS and edge.target_hash not in visited:
                    visited.add(edge.target_hash)
                    queue.append((edge.target_hash, path + [edge.action_id]))

        return None

    def is_known_dead(self, frame_hash: str, action_id: int) -> bool:
        """Check if a specific action from a state is known to be dead."""
        node = self.nodes.get(frame_hash)
        if node is None:
            return False
        if action_id >= len(node.edges):
            return False
        return node.edges[action_id].result == EdgeResult.DEAD

    def get_stats(self) -> dict:
        """Return graph statistics."""
        total_nodes = len(self.nodes)
        total_edges = sum(len(n.edges) for n in self.nodes.values())
        tested = sum(
            1 for n in self.nodes.values()
            for e in n.edges if e.result != EdgeResult.UNTESTED
        )
        dead = sum(
            1 for n in self.nodes.values()
            for e in n.edges if e.result == EdgeResult.DEAD
        )
        return {
            "nodes": total_nodes,
            "edges": total_edges,
            "tested": tested,
            "dead": dead,
            "untested": total_edges - tested,
        }

    def reset(self) -> None:
        """Reset for a new level."""
        self.nodes.clear()
        self.level_start_hash = None
