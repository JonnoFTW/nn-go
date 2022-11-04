package layers

import (
	"log"
	"nn-go/nn"
)

type Dense struct {
	inputs          int
	units           int
	weights         *nn.Matrix
	activator       nn.Activator
	initializer     nn.Initializer
	biasInitializer nn.Initializer
	useBias         bool
	biases          *nn.Matrix
	learning        bool
}

func (l *Dense) Init(inputs int) int {
	l.inputs = inputs
	l.weights = nn.NewMatrix(inputs, l.units)
	l.weights.Initialize(l.initializer, l)
	if l.useBias {
		l.biases = nn.NewMatrix(1, l.units)
		l.biases.Initialize(l.biasInitializer, l)
	}
	return l.units
}

func (l *Dense) Forward(input *nn.Matrix) *nn.Matrix {
	result := input.Product(l.weights)
	if l.useBias {
		result.Add(l.biases)
	}
	result.ActivateInPlace(l.activator)
	return result
}

// Backward pass through the network, updating if learning enabled
func (l *Dense) Backward(input *nn.Matrix) *nn.Matrix {
	return input
}

func (l *Dense) Inputs() int {
	return l.inputs
}
func (l *Dense) Outputs() int {
	return l.units
}

func NewDenseLayer(
	units int,
	useBias bool,
	activator nn.Activator,
	initializer nn.Initializer,
	biasInitializer nn.Initializer) *Dense {
	if units < 1 {
		log.Fatalf("Layers units must be more than 1, got %d\n", units)
	}
	var biases *nn.Matrix
	var weights *nn.Matrix

	return &Dense{
		0,
		units,
		weights,
		activator,
		initializer,
		biasInitializer,
		useBias,
		biases,
		true,
	}
}
