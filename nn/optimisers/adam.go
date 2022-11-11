package optimisers

import (
	math "github.com/chewxy/math32"
	"nn-go/nn"
)

type Adam struct {
	alpha   float32
	beta1   float32
	beta2   float32
	epsilon float32
}

func NewAdamOptimizer() *Adam {
	return &Adam{
		alpha:   0.001,
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1. / 1e8,
	}
}
func (a *Adam) Call(layer nn.Layer) float32 {
	return 0
}

func (a *Adam) Update(epoch int, _ nn.TrainingResults) {
	a.alpha = a.alpha / math.Sqrt(float32(epoch))
}

func (a *Adam) Lr() float32 {
	return a.alpha
}
