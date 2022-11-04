package nn

type Layer interface {
	Init(inputs int) int
	Forward(input *Matrix) *Matrix
	Backward(input *Matrix) *Matrix
	Inputs() int
	Outputs() int
}
