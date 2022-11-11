package layers

import (
	"nn-go/nn"
	"nn-go/nn/activations"
)

type Relu struct {
	units int
}

func (l *Relu) Init(inputs int) int {
	l.units = inputs
	return l.units
}

func (l *Relu) Forward(input *nn.Matrix) *nn.Matrix {
	result := input.Copy()
	result.ActivateInPlace(activations.ReLU)
	return result
}

// Backward pass through the network, updating if learning enabled
func (l *Relu) Backward(input *nn.Matrix, grads *nn.Matrix, optimizer *nn.Optimizer) *nn.Matrix {
	reluGrad := input.NonZero()
	return reluGrad.Mult(grads)
}

func NewReluLayer() *Relu {
	return &Relu{0}
}
