import random
from engine import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        activation = sum( xi*wi for xi, wi in zip(x, self.w)) + self.b
        return activation.relu() if self.nonlin else activation
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neuron({len(self.w)})"
    
class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        return [parameters for neuron in self.neurons for parameters in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
    def __init__(self, nin, nouts):
        mlp = [nin] + nouts
        self.layers = [Layer(mlp[n], mlp[n+1], nonlin=n!=len(nouts)-1) for n in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def parameters(self):
        return [parameters for layer in self.layers for parameters in layer.parameters()]
    
    def zero_grad(self):
        for parameter in self.parameters():
        	    parameter.grad = 0.0
    
