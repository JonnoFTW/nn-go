package loss

import (
	math "github.com/chewxy/math32"
	"nn-go/nn"
)

type CategoricalCrossEntropy struct{}

// Call will return a column vector of losses for the batch
func (c *CategoricalCrossEntropy) Call(observed *nn.Matrix, expected *nn.Matrix) *nn.Matrix {
	rows, classes := expected.Shape()
	losses := nn.NewMatrix(rows, 1)
	// both should be a single row of outputs
	for i := 0; i < rows; i++ {
		var sum float32
		for j := 0; j < classes; j++ {
			sum += expected.Get(i, j) * math.Log2(observed.Get(i, j)+0.00001)
		}
		losses.Set(i, 0, -sum)
	}
	return losses
}

// Gradient of observations
func (c *CategoricalCrossEntropy) Gradient(observed *nn.Matrix, expected *nn.Matrix) *nn.Matrix {
	grads := nn.NewMatrixLike(observed)
	//grads

	observed.Add(expected).Divn(1)
	return grads
}
