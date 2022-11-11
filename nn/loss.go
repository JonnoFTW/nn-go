package nn

type Loss interface {
	Call(observed *Matrix, expected *Matrix) *Matrix
	Gradient(loss *Matrix, expected *Matrix) *Matrix
}
