package nn

import (
	"nn-go/nn/activations"
	"nn-go/nn/initializers"
	"nn-go/nn/layers"
	"nn-go/nn/matrix"
	"testing"
)

func TestModel(t *testing.T) {
	model := NewModel(2)
	model.AddLayer(layers.NewDenseLayer(8, true, activations.ReLU, initializers.He, initializers.Epsilon))
	model.AddLayer(layers.NewDenseLayer(4, true, activations.ReLU, initializers.He, initializers.Epsilon))
	model.AddLayer(layers.NewDenseLayer(2, true, activations.ReLU, initializers.He, initializers.Epsilon))
	model.AddLayer(layers.NewDenseLayer(4, true, activations.ReLU, initializers.He, initializers.Epsilon))
	model.AddLayer(layers.NewDenseLayer(8, true, activations.ReLU, initializers.He, initializers.Epsilon))
	model.AddLayer(layers.NewDenseLayer(1, true, activations.Sigmoid, initializers.Glorot, initializers.Zero))
	model.Init()
	inputs := matrix.MatrixFromArray([][]float64{
		{0.888, 0.490},
	})
	output := model.Forward(inputs)
	output.Print()
}
