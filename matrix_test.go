package nn_go

import (
	"testing"
)

func TestSum(t *testing.T) {
	a := NewMatrix(4, 4)
	a.Fill(5)
	sum := a.Sum()
	if sum != 4*4*5 {
		t.Errorf("Sum was incorrect, got :%.2f, want: %d", sum, 4*4*5)
	}
}

func TestMatrix_Product(t *testing.T) {
	a := NewMatrix(4, 3)
	b := NewMatrix(3, 3)
	a.v = [][]float64{
		{1, 0, 1},
		{2, 1, 1},
		{0, 1, 1},
		{1, 1, 2},
	}
	b.v = [][]float64{
		{1, 2, 1},
		{2, 3, 1},
		{4, 2, 2},
	}
	prod := a.Product(b)
	prod.Print()
	ab := NewMatrix(4, 3)
	ab.v = [][]float64{
		{5, 4, 3},
		{8, 9, 5},
		{6, 5, 3},
		{11, 9, 6},
	}
	if !prod.Eq(ab).All() {
		t.Fatal("Invalid matrix multiplication, got")
	}

}