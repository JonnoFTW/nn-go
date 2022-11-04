package layers

import (
	"log"
	"nn-go/nn/activations"
	"nn-go/nn/initializers"
	"nn-go/nn/matrix"
)

type Dense struct {
	units           int
	weights         *matrix.Matrix
	activator       activations.Activator
	initializer     initializers.Initializer
	biasInitializer initializers.Initializer
	useBias         bool
	biases          *matrix.Matrix
	learning        bool
}

func (l *Dense) Init(inputs int) int {
	l.weights = matrix.NewMatrix(inputs, l.units)
	l.weights.Initialize(l.initializer, inputs, l.units)
	if l.useBias {
		l.biases = matrix.NewMatrix(1, l.units)
		l.biases.Initialize(l.biasInitializer, inputs, l.units)
	}
	return l.units
}

func (l *Dense) Forward(input *matrix.Matrix) *matrix.Matrix {
	result := input.Product(l.weights)
	if l.useBias {
		result.Add(l.biases)
	}
	result.ActivateInPlace(l.activator)
	return result
}

// Backward pass through the network, updating if learning enabled
func (l *Dense) Backward(input *matrix.Matrix) *matrix.Matrix {
	return input
}

func NewDenseLayer(
	units int,
	useBias bool,
	activator activations.Activator,
	initializer initializers.Initializer,
	biasInitializer initializers.Initializer) *Dense {
	if units < 1 {
		log.Fatalf("Layers units must be more than 1, got %d\n", units)
	}
	var biases *matrix.Matrix
	var weights *matrix.Matrix

	return &Dense{
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
