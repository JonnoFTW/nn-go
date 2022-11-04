package layers

import (
	"nn-go/nn/matrix"
)

type Layer interface {
	Init(inputs int) int
	Forward(input *matrix.Matrix) *matrix.Matrix
	Backward(input *matrix.Matrix) *matrix.Matrix
}
