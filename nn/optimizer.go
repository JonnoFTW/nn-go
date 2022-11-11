package nn

type Optimizer interface {
	Call(layer Layer) float32
	Update(epoch int, results TrainingResults)
	Lr() float32
}
