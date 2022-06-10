package activations

import "math"

type Activator func(float64) float64

func ReLU(v float64) float64 {
	return math.Max(0, v)
}

func Linear(v float64) float64 {
	return v
}

func Sigmoid(v float64) float64 {
	return 1. / (1. + math.Exp(-v))
}
