package nn

import (
	"log"
	"nn-go/nn/layers"
	"nn-go/nn/matrix"
)

type Model struct {
	inputs      int
	layers      []layers.Layer
	initialized bool
}

func NewModel(inputs int) *Model {
	var layers_ []layers.Layer
	return &Model{
		inputs,
		layers_,
		false,
	}
}

func (m *Model) AddLayer(layer layers.Layer) layers.Layer {
	m.layers = append(m.layers, layer)
	return layer
}

// Init initialise the model weights
func (m *Model) Init() {
	if len(m.layers) == 0 {
		log.Fatal("Model must have at least 1 layer")
	}
	inputs := m.inputs
	for _, l := range m.layers {
		inputs = l.Init(inputs)
	}
	m.initialized = true
}

// Forward calculate the forward pass through the network and calculate the final output
func (m *Model) Forward(inputs *matrix.Matrix) *matrix.Matrix {
	if !m.initialized {
		log.Fatal("Must call Init() before Forward()")
	}
	activations := inputs.Copy()
	for _, l := range m.layers {
		activations = l.Forward(activations)
	}
	return activations
}

// Loss calculate the loss and gradients
func (m *Model) Loss(vals *matrix.Matrix, target *matrix.Matrix) *matrix.Matrix {
	return matrix.NewMatrix(1, 1)
}

// Backward calculate the backward pass and update the weights
func (m *Model) Backward(grads *matrix.Matrix) {

}

// Train the model.
//  trainX - the
func (m *Model) Train(trainX *matrix.Matrix, trainY *matrix.Matrix, validation *matrix.Matrix, epochs int) {

	for i := 0; i < epochs; i++ {

	}
}
