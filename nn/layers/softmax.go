package layers

import (
	"nn-go/nn"
)

type Softmax struct {
	inputs  int
	outputs int
}

func (l *Softmax) Inputs() int {
	return l.inputs
}

func (l *Softmax) Outputs() int {
	return l.outputs
}

func (l *Softmax) Init(inputs int) int {
	l.inputs = inputs
	return l.outputs
}

func (l *Softmax) Forward(input *nn.Matrix) *nn.Matrix {
	return input.Softmax()
}

// Backward pass through the network, updating if learning enabled
func (l *Softmax) Backward(input *nn.Matrix, grads *nn.Matrix, optimizer nn.Optimizer) *nn.Matrix {
	// The softmax gradient is
	for i := 0; i < input.Rows(); i++ {
		//for j := 0; j < ; j++ {
		//
		//}
	}
	return grads
}

func NewSoftmaxLayer(outputs int) *Softmax {
	return &Softmax{0, outputs}
}
