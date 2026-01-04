import heapq
from collections import Counter

class Node:
    def __init__(self, sym=None, freq=0, left=None, right=None):
        self.sym = sym
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoder:
    def build(self, data):
        freq = Counter(data)
        heap = [Node(s, f) for s, f in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            # get smallest two
            a = heapq.heappop(heap) 
            b = heapq.heappop(heap)
            heapq.heappush(heap, Node(freq = a.freq + b.freq, left = a, right=b))

        self.root = heap[0]
        self.codes = {}
        self._gen(self.root, "")

    def _gen(self, node, code):
        if node.sym is not None:
            self.codes[node.sym] = code
            return
        
        self._gen(node.left, code + "0")
        self._gen(node.right, code + "1")

    def encode(self, data):
        return "".join(self.codes[s] for s in data)

    def decode(self, bitstream):        
        decoded = []
        node = self.root
        
        if node.left is None and node.right is None:
            return [node.sym] * len(bitstream)

        for bit in bitstream:
            if bit == "0":
                if node.left is not None:
                    node = node.left
            else:
                if node.right is not None:
                    node = node.right

            if node.sym is not None:
                decoded.append(node.sym)
                node = self.root
        return decoded