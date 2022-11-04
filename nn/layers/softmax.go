package layers

import (
	"nn-go/nn/matrix"
)

type Softmax struct {
	units int
}

func (l *Softmax) Init(inputs int) int {
	l.units = inputs
	return l.units
}

func (l *Softmax) Forward(input *matrix.Matrix) *matrix.Matrix {
	return input.Softmax()
}

// Backward pass through the network, updating if learning enabled
func (l *Softmax) Backward(input *matrix.Matrix) *matrix.Matrix {
	return input
}

func NewSoftmaxLayer() *Softmax {
	return &Softmax{0}
}
