package nn

type Initializer interface {
	Call(layer Layer) float64
}
