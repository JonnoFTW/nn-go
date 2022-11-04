package layers

import (
	"nn-go/nn/activations"
	"nn-go/nn/matrix"
)

type Relu struct {
	units int
}

func (l *Relu) Init(inputs int) int {
	l.units = inputs
	return l.units
}

func (l *Relu) Forward(input *matrix.Matrix) *matrix.Matrix {
	result := input.Copy()
	result.ActivateInPlace(activations.ReLU)
	return result
}

// Backward pass through the network, updating if learning enabled
func (l *Relu) Backward(input *matrix.Matrix) *matrix.Matrix {
	return input
}

func NewReluLayer() *Relu {
	return &Relu{0}
}
