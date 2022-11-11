package nn

type Layer interface {
	Init(inputs int) int
	Forward(input *Matrix) *Matrix
	Backward(input *Matrix, grads *Matrix, optimizer Optimizer) *Matrix
	Inputs() int
	Outputs() int
}
