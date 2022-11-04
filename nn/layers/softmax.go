package layers

import (
	"nn-go/nn"
)

type Softmax struct {
	units int
}

func (l *Softmax) Init(inputs int) int {
	l.units = inputs
	return l.units
}

func (l *Softmax) Forward(input *nn.Matrix) *nn.Matrix {
	return input.Softmax()
}

// Backward pass through the network, updating if learning enabled
func (l *Softmax) Backward(input *nn.Matrix) *nn.Matrix {
	return input
}

func NewSoftmaxLayer() *Softmax {
	return &Softmax{0}
}
