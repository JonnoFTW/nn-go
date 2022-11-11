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
	a := nn.NewMatrixFromArray([][]float32{
		{1, 0, 1},
		{2, 1, 1},
		{0, 1, 1},
		{1, 1, 2},
	})
	b := nn.NewMatrixFromArray([][]float32{
		{1, 2, 1},
		{2, 3, 1},
		{4, 2, 2},
	})
	prod := a.Product(b)
	prod.Print()
	ab := nn.NewMatrixFromArray([][]float32{
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
	a := nn.NewMatrixFromArray([][]float32{
		{4, 0},
		{0, 4},
	})
	b := nn.NewMatrixFromArray([][]float32{
		{4, 0, 0},
		{0, 0, 4},
	})
	prod := a.Product(b)
	prod.Print()
	ab := nn.NewMatrixFromArray([][]float32{
		{16, 0, 0},
		{0, 0, 16},
	})
	if !prod.Eq(ab).All() {
		t.Fatal("Invalid matrix multiplication, got")
	}

}

func TestMatrix_MeanCols(t *testing.T) {
	a := nn.NewMatrixFromArray([][]float32{
		{2, 3, 4},
		{5, 6, 7},
	})

	aMeanCols := a.MeanCols()

	expectedMeanCols := nn.NewMatrixFromArray([][]float32{
		{3.5, 4.5, 5.5},
	})

	if !aMeanCols.Eq(expectedMeanCols).All() {
		t.Fatal("Invalid col mean. Got", aMeanCols, "expected", expectedMeanCols)
	}
}

func TestMatrix_MeanRows(t *testing.T) {
	a := nn.NewMatrixFromArray([][]float32{
		{2, 3, 4},
		{5, 6, 7},
	})
	aMeanRows := a.MeanRows()
	expectedMeanRows := nn.NewMatrixFromArray([][]float32{
		{3.0, 6.0},
	})

	if !aMeanRows.Eq(expectedMeanRows).All() {
		t.Fatal("Invalid row mean. Got", aMeanRows, "expected", expectedMeanRows)
	}
}

func TestMatrix_SumRows(t *testing.T) {
	a := nn.NewMatrixFromArray([][]float32{
		{2, 3, 4},
		{5, 6, 7},
	})
	aSumRows := a.SumRows()
	expectedSumRows := nn.NewMatrixFromArray([][]float32{
		{9.0, 8.0},
	})

	if !aSumRows.Eq(expectedSumRows).All() {
		t.Fatal("Invalid row mean. Got", aSumRows, "expected", expectedSumRows)
	}
}
