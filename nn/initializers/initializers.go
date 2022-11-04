package initializers

import (
	"math"
	"math/rand"
	"nn-go/nn"
)

func uniformInRange(low float64, high float64) float64 {
	return low + rand.Float64()*(high-low)
}

type Glorot struct{}
type He struct{}
type Lecun struct{}
type Zero struct{}
type Const struct {
	val float64
}

func (g Glorot) Call(layer nn.Layer) float64 {
	low := -(math.Sqrt(6) / math.Sqrt(float64(layer.Inputs()+layer.Outputs())))
	high := -low
	return uniformInRange(low, high)
}

func (h He) Call(layer nn.Layer) float64 {
	limit := math.Sqrt(6. / float64(layer.Inputs()))
	return uniformInRange(-limit, limit)
}

func (l Lecun) Call(layer nn.Layer) float64 {
	limit := math.Sqrt(3 / float64(layer.Inputs()))
	return uniformInRange(-limit, limit)
}

func (z Zero) Call(nn.Layer) float64 {
	return 0.
}

func (c Const) Call(nn.Layer) float64 {
	return c.val
}

func NewConstInitializer(val float64) Const {
	return Const{val}
}
