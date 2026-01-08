import tensor

class Node:
    def __init__(self, data, _children=(), _op=''):
        if not isinstance(data, tensor.FloatTensor):
            raise ValueError("Data argument type must be FloatTensor")

        self.data = data
        self._prev = set(_children)
        self._op = _op

        self.grad = None
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Node):
            raise ValueError("Invalid argument data type")

        out = Node(self.data + other.data, (self, other), '+')

        def _backward():
            if self.grad is None:
                self.grad = out.grad
            else:
                self.grad += out.grad

            if other.grad is None:
                other.grad = out.grad
            else:
                other.grad += out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        if not isinstance(other, Node):
            raise ValueError("Invalid argument data type")

        out = Node(self.data @ other.data, (self, other), '@')

        def _backward():
            if self.grad is None:
                self.grad = out.grad @ tensor.FloatTensor.transpose(other.data, 0, 1)
            else:
                self.grad += out.grad @ tensor.FloatTensor.transpose(other.data, 0, 1)

            if other.grad is None:
                other.grad = tensor.FloatTensor.transpose(self.data, 0, 1) @ out.grad
            else:
                other.grad += tensor.FloatTensor.transpose(self.data, 0, 1) @ out.grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"""
                
        Node(shape={self.data.getShape()},
            op='{self._op}', 
            prev='{self._prev}',
            grad='{self.grad}')
            """

def topoSort(node, result, visited):
    if node not in visited:
        visited.add(node)
        for next in node._prev:
           topoSort(next, result, visited)
        result.append(node)

def backward(start):
    nodes = []
    visited = set()
    topoSort(start, nodes, visited)

    for node in reversed(nodes):
        node._backward()

