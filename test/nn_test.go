package test

import (
	"nn-go/nn"
	"nn-go/nn/activations"
	"nn-go/nn/initializers"
	"nn-go/nn/layers"
	"nn-go/nn/loss"
	"nn-go/nn/optimisers"
	"testing"
)

func TestModel(t *testing.T) {
	model := nn.NewModel(2, &loss.CategoricalCrossEntropy{}, optimisers.NewAdamOptimizer())
	constInitializer := initializers.NewConstInitializer(0.01)
	model.AddLayer(layers.NewDenseLayer(8, true, activations.ReLU, initializers.He{}, constInitializer))
	model.AddLayer(layers.NewDenseLayer(4, true, activations.ReLU, initializers.He{}, constInitializer))
	model.AddLayer(layers.NewDenseLayer(2, true, activations.ReLU, initializers.He{}, constInitializer))
	model.AddLayer(layers.NewDenseLayer(4, true, activations.ReLU, initializers.He{}, constInitializer))
	model.AddLayer(layers.NewDenseLayer(8, true, activations.ReLU, initializers.He{}, constInitializer))
	model.AddLayer(layers.NewDenseLayer(1, true, activations.Sigmoid, initializers.Glorot{}, initializers.Zero{}))
	model.Init()
	inputs := nn.NewMatrixFromArray([][]float32{
		{0.888, 0.490},
	})
	output := model.Predict(inputs)
	output.Print()
}
