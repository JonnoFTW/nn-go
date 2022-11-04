package nn_go

import "testing"

func TestModel(t *testing.T) {
	model := NewModel(2)
	model.AddLayer(8, true, ReLU, He, Epsilon)
	model.AddLayer(4, true, ReLU, He, Epsilon)
	model.AddLayer(2, true, ReLU, He, Epsilon)
	model.AddLayer(4, true, ReLU, He, Epsilon)
	model.AddLayer(8, true, ReLU, He, Epsilon)
	model.AddLayer(1, true, Sigmoid, Glorot, Zero)
	model.Init()
	inputs := MatrixFromArray([][]float64{
		{0.888, 0.490},
	})
	output := model.Forward(inputs)
	output.Print()
}
