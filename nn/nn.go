package nn

import (
	"fmt"
	"log"
	"math/rand"
	"time"
)

type Model struct {
	inputs      int
	layers      []Layer
	initialized bool
	loss        Loss
	optimizer   Optimizer
}

func NewModel(inputs int, loss Loss, optimizer Optimizer) *Model {
	var layers_ []Layer
	return &Model{
		inputs,
		layers_,
		false,
		loss,
		optimizer,
	}
}

func (m *Model) AddLayer(layer Layer) *Model {
	m.layers = append(m.layers, layer)
	return m
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
// return the activations of the input and each layer
func (m *Model) Forward(inputs *Matrix) []*Matrix {
	var layerActivations []*Matrix
	if !m.initialized {
		log.Fatal("Must call Init() before Forward()")
	}
	activations := inputs.Copy()
	layerActivations = append(layerActivations, activations)
	for _, l := range m.layers {
		activations = l.Forward(activations)
		layerActivations = append(layerActivations, activations)
	}
	return layerActivations
}

// Loss calculate the loss and gradients
func (m *Model) Loss(predictions *Matrix, target *Matrix) *Matrix {
	return m.loss.Call(predictions, target)
}

// LossGrads loss gradients
func (m *Model) LossGrads(predictions *Matrix, y *Matrix) *Matrix {
	return m.loss.Gradient(predictions, y)
}

// Backward calculate the backward pass and update the weights
func (m *Model) Backward(activations []*Matrix, grads *Matrix) {
	for i := len(m.layers); i != 0; i++ {
		grads = m.layers[i].Backward(activations[len(m.layers)-i], grads, m.optimizer)
	}
}

// Predict based off the input
func (m *Model) Predict(inputs *Matrix) *Matrix {
	activations := m.Forward(inputs)
	return activations[len(activations)-1]
}

type DataSet struct {
	Instances *Matrix
	Labels    *Matrix
}

func (d *DataSet) getBatch(size int, idx int) (*Matrix, *Matrix) {
	return d.Instances.Batch(size, idx), d.Labels.Batch(size, idx)
}

type TrainTestSet struct {
	Train DataSet
	Test  DataSet
}

func (d *DataSet) shuffle() {
	for i := range d.Instances.v {
		j := rand.Intn(i + 1)
		d.Instances.v[i], d.Instances.v[j] = d.Instances.v[j], d.Instances.v[i]
		d.Labels.v[i], d.Labels.v[j] = d.Labels.v[j], d.Labels.v[i]
	}
}

type TrainArgs struct {
	data              *TrainTestSet
	validation        *Matrix
	epochs            int
	batchSize         int
	shuffleAfterEpoch bool
}

func NewTrainArgs(tts *TrainTestSet, validation *Matrix, epochs int, batchSize int, shuffle bool) *TrainArgs {
	return &TrainArgs{
		tts,
		validation,
		epochs,
		batchSize,
		shuffle,
	}
}

type TrainingResults struct {
	testLosses  []float32
	trainLosses []float32
}

// Train the model.
func (m *Model) Train(args *TrainArgs) {
	if args.data.Train.Instances.rows != args.data.Train.Labels.rows {
		log.Fatalf("Number of training instances does match number of labels (%d != %d)",
			args.data.Train.Instances.rows, args.data.Train.Labels.rows)
	}
	trainSamplesCount := args.data.Train.Instances.rows
	if remainder := trainSamplesCount % args.batchSize; remainder != 0 {
		log.Fatalf("Cannot split matrix into batches of size %d. Decrease batchSize by %d or increase by %d ",
			args.batchSize, remainder, args.batchSize-remainder)
	}
	totalBatches := trainSamplesCount / args.batchSize

	var testLosses []float32
	var trainLosses []float32
	results := TrainingResults{
		testLosses:  testLosses,
		trainLosses: trainLosses,
	}
	for i := 1; i <= args.epochs; i++ {
		epochStart := time.Now().UnixMilli()
		epochLossTotal := float32(0)
		for batchIdx := 0; batchIdx < totalBatches; batchIdx++ {
			batchX, batchY := args.data.Train.getBatch(args.batchSize, batchIdx)
			layerActivations := m.Forward(batchX)
			predictions := layerActivations[len(layerActivations)-1]
			batchLoss := m.Loss(predictions, batchY).Mean()
			epochLossTotal += batchLoss
			fmt.Printf("Batch %d activations=%d loss=%.3f\n", batchIdx, len(layerActivations), batchLoss)
			lossGrads := m.LossGrads(predictions, batchY)
			m.Backward(layerActivations, lossGrads)
		}
		batchLossMean := epochLossTotal / float32(totalBatches)
		results.testLosses = append(results.testLosses, batchLossMean)

		// Evaluate performance, put it in array
		testX, testY := args.data.Test.Instances, args.data.Test.Labels
		predictions := m.Predict(testX)

		testLoss := m.Loss(predictions, testY).Mean()
		testLosses = append(testLosses, testLoss)
		// Shuffle if we want
		if args.shuffleAfterEpoch {
			args.data.Train.shuffle()
		}
		m.optimizer.Update(i, results)
		epochEnd := time.Now().UnixMilli()
		log.Printf("Epoch %d (%dms)", i, epochEnd-epochStart)
	}
}
