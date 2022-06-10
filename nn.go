package nn_go

import (
	"log"
	"nn-go/activations"
)

type Model struct {
	layers []Layer
}

type Layer struct {
	weights   *Matrix
	activator activations.Activator
	useBias   bool
	biases    Matrix
}

func (m *Model) AddLayer(units int, useBias bool, activator activations.Activator) Layer {
	if units <= 0 {
		log.Fatal("Layer size must be more than 0")
	}
	weights := NewMatrix(units, 1)
	//biases
	if activator == nil {
		// set it to He or Xavier uniform
	}
	layer := Layer{weights, activator, useBias, biases}
	return layer
}
