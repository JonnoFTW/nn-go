package nn_go

import (
	"fmt"
	"log"
)

type Matrix struct {
	rows int
	cols int
	v    [][]float64
}

func NewMatrix(rows int, cols int) *Matrix {
	if rows <= 0 || cols <= 0 {
		log.Fatal("Cannot have negative columns or rows")
	}
	vals := make([][]float64, rows)
	for i := range vals {
		vals[i] = make([]float64, cols)
	}
	return &Matrix{rows, cols, vals}
}

// Fill a matrix with the same value v
func (m *Matrix) Fill(v float64) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] = v
		}
	}
	return m
}

type initializer func() float64

// Initialize a matrix with values by calling fn
func (m *Matrix) Initialize(fn initializer) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] = fn()
		}
	}
	return m
}

// check if 2 matrices have the same dimensions, crash if they do not
func (m *Matrix) check(n *Matrix) {
	if m.rows != n.rows || m.cols != n.cols {
		log.Fatal("Array dimensions don't match")
	}
}

// Add the other matrix into this matrix, in-place
func (m *Matrix) Add(n *Matrix) *Matrix {
	m.check(n)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < n.rows; j++ {
			m.v[i][j] += n.v[i][j]
		}
	}
	return m
}

// Sub the other matrix into this matrix, in-place
func (m *Matrix) Sub(n *Matrix) *Matrix {
	m.check(n)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < n.rows; j++ {
			m.v[i][j] -= n.v[i][j]
		}
	}
	return m
}

// Eq return 2d matrix of 0 or 1 indicating where the values match
func (m *Matrix) Eq(n *Matrix) *Matrix {
	m.check(n)
	out := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < n.cols; j++ {
			e := 0.0
			if m.v[i][j] == n.v[i][j] {
				e = 1
			}
			out.v[i][j] = e
		}
	}
	return out
}

// Sum of all elements in array
func (m *Matrix) Sum() float64 {
	sum := 0.0
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			sum += m.v[i][j]
		}
	}
	return sum
}

// All returns true if all elements are 1
func (m *Matrix) All() bool {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if m.v[i][j] != 1.0 {
				return false
			}
		}
	}
	return true
}

// Some returns true if some elements are 1
func (m *Matrix) Some() bool {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if m.v[i][j] == 1.0 {
				return true
			}
		}
	}
	return false
}

// Product Calculates the matrix product
// If A of N rows, M cols
//    B of N rows, P cols
// AB=C of N rows and P cols
func (m *Matrix) Product(n *Matrix) *Matrix {
	N := m.rows
	M := m.cols
	P := n.cols

	if m.cols != n.rows {
		log.Fatal("Cannot calculate matrix product as matrix dimensions are incorrect")
	}

	out := NewMatrix(N, P)
	for i := 0; i < N; i++ {
		for j := 0; j < P; j++ {
			sum := 0.0
			for k := 0; k < M; k++ {
				sum += m.v[i][k] * n.v[k][j]
			}
			out.v[i][j] = sum
		}
	}
	return out
}

// T calculates the transpose of a matrix and returns it as a copy
func (m *Matrix) T() *Matrix {
	out := NewMatrix(m.cols, m.rows)
	for i := 0; i < out.rows; i++ {
		for j := 0; j < out.cols; j++ {
			out.v[i][j] = m.v[j][i]
		}
	}
	return out
}

// Print the matrix nicely
func (m *Matrix) Print() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			fmt.Printf("%.2f ", m.v[i][j])
		}
		fmt.Println()
	}
}
