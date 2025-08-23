import random


class Node:
    def __init__(self, name):
        self.name = name
        self.known_nodes = set([name])

    def gossip_with(self, other_node):
        self.known_nodes |= other_node.known_nodes
        other_node.known_nodes |= self.known_nodes


def simulate_gossip(node_count, seed_nodes=None, rounds=5):
    nodes = {f"N{i}": Node(f"N{i}") for i in range(node_count)}

    if seed_nodes:
        for name in seed_nodes:
            nodes[name].known_nodes |= set(seed_nodes)

    for round_num in range(1, rounds + 1):
        for node in nodes.values():
            if seed_nodes and node.name not in seed_nodes:
                choices = list(seed_nodes) * 2 + list(nodes.keys())
            else:
                choices = list(nodes.keys())

            choices = [n for n in choices if n != node.name]
            partner = nodes[random.choice(choices)]
            node.gossip_with(partner)

        print(f"After round {round_num}")
        for n in nodes.values():
            print(f" {n.name} knows {len(n.known_nodes)}/{node_count} nodes -> {sorted(n.known_nodes)}")
        print("-" * 40)


# Example without seeds
print("=== Without Seeds ===")
simulate_gossip(node_count=10, seed_nodes=None, rounds=5)

print("===With Seeds===")
simulate_gossip(node_count=10, seed_nodes=["N0", "N1", "N2"], rounds=5)
