package nn_go

import (
	"math"
	"math/rand"
)

func uniformInRange(low float64, high float64) float64 {
	return low + rand.Float64()*(high-low)
}

type Initializer func(int, int) float64

func Glorot(inputs int, outputs int) float64 {
	low := -(math.Sqrt(6) / math.Sqrt(float64(inputs+outputs)))
	high := -low
	return uniformInRange(low, high)
}

func He(inputs int, outputs int) float64 {
	limit := math.Sqrt(6. / float64(inputs))
	return uniformInRange(-limit, limit)
}

func Lecun(inputs int, outputs int) float64 {
	limit := math.Sqrt(3 / float64(inputs))
	return uniformInRange(-limit, limit)
}

func Zero(inputs int, outputs int) float64 {
	return 0.
}

func Epsilon(inputs int, output int) float64 {
	return 0.01
}
