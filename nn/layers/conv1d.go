package layers

import (
	"log"
	"nn-go/nn"
)

type Conv1d struct {
	inputs          int
	filters         int
	kernelSize      int
	strides         int
	padding         Padding
	weights         *nn.Matrix
	activator       nn.Activator
	initializer     nn.Initializer
	biasInitializer nn.Initializer
	useBias         bool
	biases          *nn.Matrix
	learning        bool
}

func (l *Conv1d) Init(inputs int) int {
	l.filters = inputs
	return inputs
	//l.weights = nn.NewMatrix(inputs, l.units)
	//l.weights.Initialize(l.initializer, l)
	//if l.useBias {
	//	l.biases = nn.NewMatrix(1, l.units)
	//	l.biases.Initialize(l.biasInitializer, l)
	//}
	//return l.units
}

func (l *Conv1d) Forward(input *nn.Matrix) *nn.Matrix {
	result := input.Product(l.weights)
	if l.useBias {
		result.Add(l.biases)
	}
	result.ActivateInPlace(l.activator)
	return result
}

// Backward pass through the network, updating weights if learning enabled
func (l *Conv1d) Backward(input *nn.Matrix, grads *nn.Matrix, optimizer *nn.Optimizer) *nn.Matrix {
	return grads
}

func (l *Conv1d) Inputs() int {
	return l.inputs
}
func (l *Conv1d) Outputs() int {
	return 0 //l.units
}

type Padding int32

const (
	VALID int = 0 // Padding is a copy
	SAME  int = 1 // Padding is 0
)

func NewConv1dLayer(
	filters int,
	kernelSize int,
	strides int,
	padding Padding,
	useBias bool,
	activator nn.Activator,
	initializer nn.Initializer,
	biasInitializer nn.Initializer) *Conv1d {
	if filters < 1 {
		log.Fatalf("Layers filters must be more than 1, got %d\n", filters)
	}
	var biases *nn.Matrix
	var weights *nn.Matrix

	return &Conv1d{
		0,
		filters,
		kernelSize,
		strides,
		padding,
		weights,
		activator,
		initializer,
		biasInitializer,
		useBias,
		biases,
		true,
	}
}
