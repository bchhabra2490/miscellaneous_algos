import hashlib
import bisect


def hash_fn(key):
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**32)


class ConsistentHashRing:
    def __init__(self, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []

    def add_node(self, node):
        for i in range(self.replicas):
            vnode_key = f"{node}#{i}"
            pos = hash_fn(vnode_key)
            self.ring[pos] = node
            bisect.insort(self.sorted_keys, pos)

    def remove_node(self, node):
        for i in range(self.replicas):
            vnode_key = f"{node}#{i}"
            pos = hash_fn(vnode_key)
            if pos in self.ring:
                del self.ring[pos]
                self.sorted_keys.remove(pos)

    def get_node(self, key):
        if not self.ring:
            return None
        pos = hash_fn(key)
        idx = bisect.bisect_right(self.sorted_keys, pos)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]


ring = ConsistentHashRing(replicas=5)

# Add nodes
ring.add_node("NodeA")
ring.add_node("NodeB")
ring.add_node("NodeC")

keys = ["user:alice", "user:bob", "user:charlie", "user:david"]

print("Initial mapping:")
for k in keys:
    print(f"{k} -> {ring.get_node(k)}")

# Add a new node
ring.add_node("NodeD")
print("\nAfter adding NodeD:")
for k in keys:
    print(f"{k} -> {ring.get_node(k)}")

# Remove a node
ring.remove_node("NodeB")
print("\nAfter removing NodeB:")
for k in keys:
    print(f"{k} -> {ring.get_node(k)}")
