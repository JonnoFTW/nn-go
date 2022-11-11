package nn

import (
	"fmt"
	math "github.com/chewxy/math32"
	"log"
)

type Matrix struct {
	rows int
	cols int
	v    [][]float32
}

func NewMatrix(rows int, cols int) *Matrix {
	if rows <= 0 || cols <= 0 {
		log.Fatal("Cannot have negative columns or rows")
	}
	vals := make([][]float32, rows)
	for i := range vals {
		vals[i] = make([]float32, cols)
	}
	return &Matrix{rows, cols, vals}
}
func NewMatrixFromArray(array [][]float32) *Matrix {
	out := NewMatrix(len(array), len(array[0]))
	for i := 0; i < len(array); i++ {
		if len(array[i]) != out.cols {
			log.Fatalf("Invalid number of columns at row %d. Expected %d, got %d", i, out.cols, len(array[i]))
		}
	}
	out.v = array
	return out
}

func NewMatrixLike(m *Matrix) *Matrix {
	out := NewMatrix(m.rows, m.cols)
	return out
}

func Ones(n int) *Matrix {
	vals := make([][]float32, n)
	for i := range vals {
		vals[i] = make([]float32, n)
		vals[i][i] = 1
	}
	return &Matrix{n, n, vals}
}

// Copy a matrix
func (m *Matrix) Copy() *Matrix {
	copyVals := make([][]float32, m.rows)
	for i := range m.v {
		copyVals[i] = make([]float32, m.cols)
		copy(copyVals[i], m.v[i])
	}
	return &Matrix{m.rows, m.cols, copyVals}
}

// Fill a matrix with the same value v
func (m *Matrix) Fill(v float32) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] = v
		}
	}
	return m
}

// Initialize a matrix with values by calling fn
func (m *Matrix) Initialize(fn Initializer, layer Layer) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] = fn.Call(layer)
		}
	}
	return m
}

// Activate apply an activation to a matrix and return a copy
func (m *Matrix) Activate(fn Activator) *Matrix {
	out := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			out.v[i][j] = fn(m.v[i][j])
		}
	}
	return out
}

// ActivateInPlace apply an activation to a matrix in-place
func (m *Matrix) ActivateInPlace(fn Activator) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] = fn(m.v[i][j])
		}
	}
	return m
}

// check if 2 matrices have the same dimensions, crash if they do not
func (m *Matrix) check(n *Matrix) {
	if m.rows != n.rows || m.cols != n.cols {
		err := fmt.Errorf("array dimensions rows,cols (%d -> %d) (%d -> %d) don't match", m.rows, n.rows, m.cols, m.cols)
		log.Fatal(err)
	}
}

// Add the other matrix into this matrix, in-place
func (m *Matrix) Add(n *Matrix) *Matrix {
	if n.rows == 1 && n.cols == m.cols {
		// Add the row matrix n to each in m
		for i := 0; i < m.rows; i++ {
			for j := 0; j < m.cols; j++ {
				m.v[i][j] += n.v[0][j]
			}
		}
	} else {
		//else if n.cols == 1 && n.rows == m.rows {
		//	// Add the column matrix n to each in m
		//	for i := 0; i < m.cols; i++ {
		//		for j := 0; j < m.rows; j++ {
		//			m.v[i][j] += n.v[j][0]
		//		}
		//	}
		//}
		m.check(n)
		for i := 0; i < m.rows; i++ {
			for j := 0; j < n.cols; j++ {
				m.v[i][j] += n.v[i][j]
			}
		}
	}
	return m
}

// Addn add a value to an array
func (m *Matrix) Addn(v float32) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] += v
		}
	}
	return m
}

// Sub the other matrix into this matrix, in-place
func (m *Matrix) Sub(n *Matrix) *Matrix {
	m.check(n)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] -= n.v[i][j]
		}
	}
	return m
}

// Subn subtract a value to an array
func (m *Matrix) Subn(v float32) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] -= v
		}
	}
	return m
}

// Div divide all values in array with values in other array, in-place
func (m *Matrix) Div(n *Matrix) *Matrix {
	m.check(n)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] /= n.v[i][j]
		}
	}
	return m
}

// Divn divide all values by v, in-place
func (m *Matrix) Divn(v float32) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] /= v
		}
	}
	return m
}

// Mult multiply all values in matrix m with those the same places in matrix n, in-place
func (m *Matrix) Mult(n *Matrix) *Matrix {
	m.check(n)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] *= n.v[i][j]
		}
	}
	return m
}

// Multn multiply all values in matrix by v, in-place
func (m *Matrix) Multn(v float32) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.v[i][j] *= v
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
			var e float32
			if m.v[i][j] == n.v[i][j] {
				e = 1
			}
			out.v[i][j] = e
		}
	}
	return out
}

// Sum of all elements in array
func (m *Matrix) Sum() float32 {
	var sum float32
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
		log.Fatalf("Cannot calculate matrix product as matrix dimensions are incorrect. m.cols should equal n.rows %d != %d", m.cols, n.rows)
	}

	out := NewMatrix(N, P)
	for i := 0; i < N; i++ {
		for j := 0; j < P; j++ {
			var sum float32
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
	fmt.Printf("Matrix with rows=%d cols=%d\n", m.rows, m.cols)
	fmt.Print("[")
	for i := 0; i < m.rows; i++ {
		fmt.Print("[ ")
		for j := 0; j < m.cols; j++ {
			fmt.Printf("%.2f, ", m.v[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Print("]\n")
}

func argMaxRow(row []float32) int {
	max := row[0]
	maxIdx := 0
	for i := 0; i < len(row); i++ {
		if row[i] > max {
			max = row[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// ArgMax the index of the maximum element in the array
func (m *Matrix) ArgMax() *Matrix {
	out := NewMatrix(m.rows, 1)
	for i := 0; i < m.rows; i++ {
		out.v[i][0] = float32(argMaxRow(m.v[i]))
	}
	return out
}

// Softmax of the matrix. Returns a new matrix
func (m *Matrix) Softmax() *Matrix {
	out := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		sum := float32(0)
		max := m.v[i][0]
		for j := 0; j < m.cols; j++ {
			if m.v[i][j] > max {
				max = m.v[i][j]
			}
		}
		for j := 0; j < m.cols; j++ {
			out.v[i][j] = math.Exp(m.v[i][j] - max)
			sum += out.v[i][j]
		}
		for j := 0; j < m.cols; j++ {
			out.v[i][j] = out.v[i][j] / sum
		}
	}
	return out
}

// Shape of the matrix
func (m *Matrix) Shape() (int, int) {
	return m.rows, m.cols
}

// Get the value at i, in the matrix
func (m *Matrix) Get(i int, j int) float32 {
	return m.v[i][j]
}

// Batch get the nth batch of a matrix. Returns a new matrix
func (m *Matrix) Batch(size int, batchIdx int) *Matrix {
	out := NewMatrix(size, m.cols)
	out.v = m.v[size*batchIdx : size*batchIdx+size]
	return out
}

// Mean the arithmetic mean of all values in the matrix
func (m *Matrix) Mean() float32 {
	return m.Sum() / float32(m.rows*m.cols)
}

// Set value at i,j in the matrix
func (m *Matrix) Set(i int, j int, val float32) {
	m.v[i][j] = val
}

// Rows the number of rows in the matrix
func (m *Matrix) Rows() int {
	return m.rows
}

// Cols the number of columns in the matrix
func (m *Matrix) Cols() int {
	return m.cols
}

// Min the minimum value in the matrix
func (m *Matrix) Min() float32 {
	min := math.Inf(1)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			min = math.Min(min, m.v[i][j])
		}
	}
	return min
}

// Max maximum value in the matrix
func (m *Matrix) Max() float32 {
	min := m.v[0][0]
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			min = math.Max(min, m.v[i][j])
		}
	}
	return min
}

// NonZero return a new matrix with 1s where the source matrix has a value > 0. Returns a new matrix
func (m *Matrix) NonZero() *Matrix {
	out := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if m.v[i][j] > 0 {
				out.v[i][j] = 1
			}
		}
	}
	return out
}

// MeanCols the mean of each column. Returns a new array
func (m *Matrix) MeanCols() *Matrix {
	out := NewMatrix(1, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			out.v[0][j] += m.v[i][j]
		}
	}
	return out.Divn(float32(m.rows))
}

//MeanRows the mean of each row. Returns a new array
func (m *Matrix) MeanRows() *Matrix {
	out := NewMatrix(1, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			out.v[0][i] += m.v[i][j]
		}
	}
	return out.Divn(float32(m.cols))
}

// SumRows the sum of each row
func (m *Matrix) SumRows() *Matrix {
	out := NewMatrix(1, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			out.v[0][i] += m.v[i][j]
		}
	}
	return out
}
