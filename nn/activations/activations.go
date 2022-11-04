package activations

import (
	"math"
)

func ReLU(v float64) float64 {
	return math.Max(0, v)
}

func Linear(v float64) float64 {
	return v
}

func Sigmoid(v float64) float64 {
	return 1. / (1. + math.Exp(-v))
}

func Softmax(v float64) float64 {
	return 1 / math.Exp(v)
}

func Tanh(v float64) float64 {
	return math.Tanh(v)
}

func LRelU(v float64) float64 {
	if v < 0 {
		return 0.01 * v
	}
	return v
}
