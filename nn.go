package nn_go

import (
	"log"
)

type Model struct {
	inputs      int
	layers      []*Layer
	initialized bool
}

type Layer struct {
	units           int
	weights         *Matrix
	activator       Activator
	initializer     Initializer
	biasInitializer Initializer
	useBias         bool
	biases          *Matrix
}

func NewModel(inputs int) *Model {
	var layers []*Layer
	return &Model{
		inputs,
		layers,
		false,
	}
}

func (m *Model) AddLayer(units int, useBias bool, activator Activator, initializer Initializer, biasInitializer Initializer) *Layer {
	if units <= 0 {
		log.Fatal("Layer size must be more than 0")
	}
	var biases *Matrix
	var weights *Matrix

	layer := &Layer{
		units,
		weights,
		activator,
		initializer,
		biasInitializer,
		useBias,
		biases,
	}
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
		l.weights = NewMatrix(inputs, l.units)
		l.weights.Initialize(l.initializer, inputs, l.units)
		if l.useBias {
			l.biases = NewMatrix(1, l.units)
			l.biases.Initialize(l.biasInitializer, inputs, l.units)
		}
		inputs = l.units
		//fmt.Printf("Layer %d has weights:\n", idx)
		//l.weights.Print()
	}
	m.initialized = true
}

// Forward calculate the forward pass through the network and calculate the final output
func (m *Model) Forward(inputs *Matrix) *Matrix {
	if !m.initialized {
		log.Fatal("Must call Init() before Forward()")
	}
	activations := inputs.Copy()
	for _, l := range m.layers {
		activations = activations.Product(l.weights)
		if l.useBias {
			activations.Add(l.biases)
		}
		activations = activations.ActivateInPlace(l.activator)
	}
	return activations
}

// Loss calculate the loss and gradients
func (m *Model) Loss(vals *Matrix, target *Matrix) *Matrix {
	return NewMatrix(1, 1)
}

// Backward calculate the backward pass and update the weights
func (m *Model) Backward(grads *Matrix) {

}

func (m *Model) Train(inputs *Matrix, epochs int) {
	for i := 0; i < epochs; i++ {

	}
}
