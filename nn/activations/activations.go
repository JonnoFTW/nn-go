package activations

import (
	math "github.com/chewxy/math32"
)

func ReLU(v float32) float32 {
	return math.Max(0, v)
}

func Linear(v float32) float32 {
	return v
}

func Sigmoid(v float32) float32 {
	return 1. / (1. + math.Exp(-v))
}

func Softmax(v float32) float32 {
	return 1 / math.Exp(v)
}

func Tanh(v float32) float32 {
	return math.Tanh(v)
}

func LRelU(v float32) float32 {
	if v < 0 {
		return 0.01 * v
	}
	return v
}
