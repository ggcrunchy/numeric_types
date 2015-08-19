--- General-purpose m-by-n matrices.

--
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
--
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
-- IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
-- CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
-- TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
-- SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--
-- [ MIT license: http://www.opensource.org/licenses/mit-license.php ]
--

-- Modules --
local assert = assert
local setmetatable = setmetatable
local sqrt = math.sqrt

-- Cached module references --
local _New_
local _Zero_

-- Exports --
local M = {}

-- --
local MatrixMethods = { __metatable = true }

MatrixMethods.__index = MatrixMethods

--
local function Index (matrix, row, col)
	assert(row >= 1 and row <= matrix.m_rows, "Bad row")
	assert(col >= 1 and col <= matrix.m_cols, "Bad column")
	
	return matrix.m_cols * (row - 1) + col
end

--
local function Set (matrix, row, col, value)
	matrix[Index(matrix, row, col)] = value
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam MatrixMN B
-- @treturn MatrixMN S
function M.Add (A, B)
	assert(A.m_rows == B.m_rows, "Mismatched rows")
	assert(A.m_cols == B.m_cols, "Mismatched columns")

	local sum = _New_(A.m_rows, A.m_cols)

	for index = 1, A.m_rows * A.m_cols do
		sum[index] = A[index] + B[index]
	end

	return sum
end

--- DOCME
-- @tparam MatrixMN A
-- @uint k1
-- @uint k2
-- @treturn MatrixMN C
function M.Columns (A, k1, k2)
	local dc, nrows = 1 - k1, A.m_rows
	local cols = _New_(nrows, k2 + dc)

	for r = 1, nrows do
		for c = k1, k2 do
			Set(cols, r, c + dc, A(r, c))
		end
	end

	return cols
end

--- DOCME
-- @tparam MatrixMN A
-- @uint row
-- @uint col
-- @treturn MatrixMN C
function M.Corner (A, row, col)
	local ncols, nrows, index = A.m_cols, A.m_rows, 1
	local from = _New_(A.m_rows - row + 1, A.m_cols - col + 1)

	for r = row, nrows do
		for c = col, ncols do
			from[index], index = A(r, c), index + 1
		end
	end

	return from
end

--- DOCME
-- @uint n
-- @treturn MatrixMN m
function M.Identity (n)
	local id = _Zero_(n, n)

	for i = 1, n * n, n + 1 do
		id[i] = 1
	end

	return id
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam MatrixMN B
-- @treturn MatrixMN P
function M.Mul (A, B)
	assert(A.m_cols == B.m_rows, "Mismatched matrices")

	local m, n, len = A.m_rows, B.m_cols, A.m_cols
	local product, index = _New_(m, n), 1

	for r = 1, m do
		for c = 1, n do
			local sum = 0

			for i = 1, len do
				sum = sum + A(r, i) * B(i, c)
			end

			product[index], index = sum, index + 1
		end		
	end

	return product
end

--- DOCME
-- @uint rows
-- @uint cols
-- @treturn MatrixMN m
function M.New (rows, cols)
	return setmetatable({ m_cols = cols, m_rows = rows }, MatrixMethods)
end

--- DOCME
-- @tparam Vector v
-- @tparam Vector w
-- @treturn MatrixMN S
function M.OuterProduct (v, w)
	local n = #v

	assert(n == #w, "Mismatched vectors")

	local outer, index = _New_(n, n), 1

	for i = 1, n do
		for j = 1, n do
			outer[index], index = v[i] * w[j], index + 1
		end
	end

	return outer
end

--- DOCME
-- @tparam MatrixMN A
-- @uint row
-- @uint col
-- @tparam MatrixMN B
function M.PutBlock (A, row, col, B)
	local row_to, ncols, index = B.m_rows + row - 1, B.m_cols, 1

	assert(row_to <= A.m_rows, "Bad row for block")
	assert(ncols + col - 1 <= A.m_cols, "Bad column for block")

	for r = row, row_to do
		local mi = Index(A, r, col)

		for at = mi, mi + ncols - 1 do
			A[at], index = B[index], index + 1
		end
	end
end

--- DOCME
-- @tparam MatrixMN A
-- @number s
-- @treturn MatrixMN S
function M.Scale (A, scale)
	local ncols, nrows = A.m_cols, A.m_rows
	local scaled = _New_(nrows, ncols)

	for i = 1, ncols * nrows do
		scaled[i] = A[i] * scale
	end

	return scaled
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam MatrixMN B
-- @treturn MatrixMN D
function M.Sub (A, B)
	assert(A.m_rows == B.m_rows, "Mismatched rows")
	assert(A.m_cols == B.m_cols, "Mismatched columns")

	local diff = _New_(A.m_rows, A.m_cols)

	for index = 1, A.m_rows * A.m_cols do
		diff[index] = A[index] - B[index]
	end

	return diff
end

--- DOCME
-- @tparam MatrixMN A
-- @treturn MatrixMN T
function M.Transpose (A)
	local nrows, ncols = A.m_rows, A.m_cols
	local trans = _New_(ncols, nrows)

	for col = 1, ncols do
		local index = col

		for _ = 1, nrows do
			trans[#trans + 1], index = A[index], index + ncols
		end
	end

	return trans
end

--- DOCME
-- @uint rows
-- @uint cols
-- @treturn MatrixMN m
function M.Zero (rows, cols)
	local matrix = _New_(rows, cols)

	for i = 1, matrix.m_rows * matrix.m_cols do
		matrix[i] = 0
	end

	return matrix
end

-- Add methods.
do
	--- Metamethod.
	-- @uint row
	-- @uint col
	-- @treturn number S
	function MatrixMethods:__call (row, col)
		return self[Index(self, row, col)]
	end

	-- GetDims()
	-- IsVector(), IsScalar()?
	-- Multiply(), TransposeMultiply()?
	-- Rank()?
	-- column, row length, length squared
	-- column, row dot products

	--- DOCME
	-- @uint col
	-- @uint[opt=1] from
	-- @treturn number L
	function MatrixMethods:ColumnLength (col, from)
		local sum, index, ncols = 0, Index(self, from or 1, col), self.m_cols

		for _ = from or 1, self.m_rows do
			sum, index = sum + self[index]^2, index + ncols
		end

		return sqrt(sum)
	end

	--- DOCME
	-- @uint col
	-- @uint[opt=1] from
	-- @treturn table C
	function MatrixMethods:GetColumn (col, from)
		local arr, index, ncols = {}, Index(self, from or 1, col), self.m_cols

		for _ = from or 1, self.m_rows do
			arr[#arr + 1], index = self[index], index + ncols
		end

		return arr
	end

	--- DOCME
	-- @uint row
	-- @uint[opt=1] from
	-- @treturn table R
	function MatrixMethods:GetRow (row, from)
		local arr, index = {}, Index(self, row, from or 1)

		for _ = from or 1, self.m_cols do
			arr[#arr + 1], index = self[index], index + 1
		end

		return arr
	end

	-- DOCME
	-- @uint row
	-- @uint[opt=1] from
	-- @treturn number L
	function MatrixMethods:RowLength (row, from)
		local sum, index = 0, Index(self, row, from or 1)

		for _ = from or 1, self.m_cols do
			sum, index = sum + self[index]^2, index + 1
		end

		return sqrt(sum)
	end

	-- ^^ TODO: Squared lengths?

	--- DOCME
	-- @function MatrixMethods:Set
	-- @uint row
	-- @uint col
	-- @number X
	MatrixMethods.Set = Set
end

-- Cache module members.
_New_ = M.New
_Zero_ = M.Zero

-- Export the module.
return M