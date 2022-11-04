package test

import (
	"nn-go/nn"
	"testing"
)

func TestSum(t *testing.T) {
	a := nn.NewMatrix(4, 4)
	a.Fill(5)
	sum := a.Sum()
	if sum != 4*4*5 {
		t.Errorf("Sum was incorrect, got :%.2f, want: %d", sum, 4*4*5)
	}
}

func TestMatrix_Product(t *testing.T) {
	a := nn.NewMatrixFromArray([][]float64{
		{1, 0, 1},
		{2, 1, 1},
		{0, 1, 1},
		{1, 1, 2},
	})
	b := nn.NewMatrixFromArray([][]float64{
		{1, 2, 1},
		{2, 3, 1},
		{4, 2, 2},
	})
	prod := a.Product(b)
	prod.Print()
	ab := nn.NewMatrixFromArray([][]float64{
		{5, 4, 3},
		{8, 9, 5},
		{6, 5, 3},
		{11, 9, 6},
	})
	if !prod.Eq(ab).All() {
		t.Fatal("Invalid matrix multiplication, got")
	}

}

func TestMatrix_Rec_Square_Product(t *testing.T) {
	a := nn.NewMatrixFromArray([][]float64{
		{4, 0},
		{0, 4},
	})
	b := nn.NewMatrixFromArray([][]float64{
		{4, 0, 0},
		{0, 0, 4},
	})
	prod := a.Product(b)
	prod.Print()
	ab := nn.NewMatrixFromArray([][]float64{
		{16, 0, 0},
		{0, 0, 16},
	})
	if !prod.Eq(ab).All() {
		t.Fatal("Invalid matrix multiplication, got")
	}

}
