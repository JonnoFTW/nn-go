package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"nn-go/nn"
	"nn-go/nn/activations"
	"nn-go/nn/initializers"
	"nn-go/nn/layers"
	"nn-go/nn/loss"
	"nn-go/nn/optimisers"
	"os"
	"strconv"
)

func makeOneHot(class int, numClass int) []float32 {
	out := make([]float32, numClass)
	out[class] = 1
	return out
}

func makeImage(row []string) []float32 {
	image := make([]float32, len(row))
	for idx, val := range row {
		intVar, err := strconv.Atoi(val)
		if err != nil {
			log.Fatal(err)
		}
		image[idx] = float32(intVar) / float32(255.0)
	}
	return image
}

func readFile(fileName string) (*nn.Matrix, *nn.Matrix) {
	trainF, err := os.Open("data/fashion-mnist/" + fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer func(trainF *os.File) {
		err := trainF.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(trainF)
	numLabels := 10
	// read csv values
	csvReader := csv.NewReader(trainF)

	var images [][]float32
	var labels [][]float32
	_, _ = csvReader.Read()
	for {
		rec, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		classId, err := strconv.Atoi(rec[0])
		if err != nil {
			log.Fatal(err)
		}
		images = append(images, makeImage(rec[1:]))
		labels = append(labels, makeOneHot(classId, numLabels))
	}

	return nn.NewMatrixFromArray(images), nn.NewMatrixFromArray(labels)
}

func load() nn.TrainTestSet {
	fmt.Println("Loading fashion mnist")
	// There are 10 labels
	trainX, trainY := readFile("fashion-mnist_train.csv")
	testX, testY := readFile("fashion-mnist_test.csv")
	xRows, xCols := trainX.Shape()
	yRows, yCols := trainY.Shape()
	fmt.Printf("\tTrain_x shape=(%d, %d) train_y shape=(%d, %d)\n", xRows, xCols, yRows, yCols)
	return nn.TrainTestSet{
		Train: nn.DataSet{
			Instances: trainX,
			Labels:    trainY,
		},
		Test: nn.DataSet{
			Instances: testX,
			Labels:    testY,
		},
	}

}

func makeModel(imageSize int) *nn.Model {
	model := nn.NewModel(
		imageSize,
		&loss.CategoricalCrossEntropy{},
		optimisers.NewAdamOptimizer(),
	)
	wHe := initializers.He{}
	bHe := initializers.NewConstInitializer(0.01)
	model.
		AddLayer(layers.NewDenseLayer(128, true, activations.ReLU, wHe, bHe)).
		AddLayer(layers.NewDenseLayer(64, true, activations.ReLU, wHe, bHe)).
		//AddLayer(layers.NewDenseLayer(32, true, activations.ReLU, wHe, bHe)).
		//AddLayer(layers.NewDenseLayer(64, true, activations.ReLU, wHe, bHe)).
		//AddLayer(layers.NewDenseLayer(128, true, activations.ReLU, wHe, bHe)).
		AddLayer(layers.NewDenseLayer(10, true, activations.Linear, wHe, bHe)).
		AddLayer(layers.NewSoftmaxLayer(10))
	model.Init()
	return model
}

func main() {
	tts := load()
	model := makeModel(tts.Train.Instances.Cols())
	model.Train(
		nn.NewTrainArgs(
			&tts,
			nil,
			2,
			4,
			true,
		),
	)
}
