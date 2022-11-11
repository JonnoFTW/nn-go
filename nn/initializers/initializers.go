package initializers

import (
	math "github.com/chewxy/math32"
	"math/rand"
	"nn-go/nn"
)

func uniformInRange(low float32, high float32) float32 {
	return low + rand.Float32()*(high-low)
}

type Glorot struct{}
type He struct{}
type Lecun struct{}
type Zero struct{}
type Const struct {
	val float32
}

func (g Glorot) Call(layer nn.Layer) float32 {
	low := -(math.Sqrt(6) / math.Sqrt(float32(layer.Inputs()+layer.Outputs())))
	high := -low
	return uniformInRange(low, high)
}

func (h He) Call(layer nn.Layer) float32 {
	limit := math.Sqrt(6. / float32(layer.Inputs()))
	return uniformInRange(-limit, limit)
}

func (l Lecun) Call(layer nn.Layer) float32 {
	limit := math.Sqrt(3 / float32(layer.Inputs()))
	return uniformInRange(-limit, limit)
}

func (z Zero) Call(nn.Layer) float32 {
	return 0.
}

func (c Const) Call(nn.Layer) float32 {
	return c.val
}

func NewConstInitializer(val float32) Const {
	return Const{val}
}
