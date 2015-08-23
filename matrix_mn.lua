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
local _Columns_From_
local _New_

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
local function NewPrep (nrows, ncols, out)
	if out then
		out:Resize(nrows, ncols)

		return out
	else
		return _New_(nrows, ncols)
	end
end

--
local function ZeroPrep (nrows, ncols, out)
	out = NewPrep(nrows, ncols, out)

	for i = 1, nrows * ncols do
		out[i] = 0
	end

	return out
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam MatrixMN B
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN S
function M.Add (A, B, out)
	local nrows, ncols = A:GetDims()

	assert(nrows == B.m_rows, "Mismatched rows")
	assert(ncols == B.m_cols, "Mismatched columns")
-- TODO: zero-pad?
	local sum = NewPrep(nrows, ncols, out)

	for index = 1, nrows * ncols do
		sum[index] = A[index] + B[index]
	end

	return sum
end

--- DOCME
-- @tparam MatrixMN A
-- @uint k1
-- @uint k2
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN C
function M.Columns (A, k1, k2, out)
	return _Columns_From_(A, k1, k2, 1, out)
end

--- DOCME
-- @tparam MatrixMN A
-- @uint k1
-- @uint k2
-- @uint from
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN C
function M.Columns_From (A, k1, k2, from, out)
	local nrows, ncols = A.m_rows, A.m_cols
	local w, skip, inc = k2 - k1 + 1

	if k2 < k1 then
		skip, inc = ncols + 1, -1
	else
		skip, inc = ncols - w, 1
	end

	out = NewPrep(nrows - from + 1, w, out)

	local index, ai = 1

	for row = from, nrows do
		ai = ai or Index(A, row, k1)

		for _ = 1, w do
			out[index], index, ai = A[ai], index + 1, ai + inc
		end

		ai = ai + skip
	end

	return out
end

--- DOCME
-- @tparam MatrixMN A
-- @uint row
-- @uint col
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN C
function M.Corner (A, row, col, out)
	local w, h = A.m_cols - col + 1, A.m_rows - row + 1

	out = NewPrep(h, w, out)

	local index, ai, skip = 1, Index(A, row, col), col - 1

	for _ = 1, h do
		for _ = 1, w do
			out[index], index, ai = A[ai], index + 1, ai + 1
		end

		ai = ai + skip
	end

	return out
end

--- DOCME
-- @tparam MatrixMN A
-- @treturn number N
function M.FrobeniusNorm (A)
	local sum = 0

	for i = 1, A.m_rows * A.m_cols do
		sum = sum + A[i]^2
	end

	return sqrt(sum)
end

--- DOCME
-- @uint n
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN m
function M.Identity (n, out)
	out = ZeroPrep(n, n, out)

	for i = 1, n * n, n + 1 do
		out[i] = 1
	end

	return out
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam MatrixMN B
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN P
function M.Mul (A, B, out)
	local m, n, len, index = A.m_rows, B.m_cols, A.m_cols, 1

	assert(len == B.m_rows, "Mismatched matrices")

	out = NewPrep(m, n, out)

	for r = 1, m do
		for c = 1, n do
			local sum = 0

			for i = 1, len do
				sum = sum + A(r, i) * B(i, c)
			end

			out[index], index = sum, index + 1
		end		
	end

	return out
end

--- DOCME
-- @uint nrows
-- @uint ncols
-- @treturn MatrixMN m
function M.New (nrows, ncols)
	return setmetatable({ m_cols = ncols, m_rows = nrows }, MatrixMethods)
end

--- DOCME
-- @tparam Vector v
-- @tparam Vector w
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN S
function M.OuterProduct (v, w, out)
	local n1, n2, index = #v, #w, 1

	out = NewPrep(n1, n2, out)

	for i = 1, n1 do
		for j = 1, n2 do
			out[index], index = v[i] * w[j], index + 1
		end
	end

	return out
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
-- @uint nrows
-- @uint ncols
function M.Resize (A, nrows, ncols)
	A.m_rows, A.m_cols = nrows, ncols
end

--- DOCME
-- @tparam MatrixMN A
-- @number s
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN S
function M.Scale (A, scale, out)
	local nrows, ncols = A.m_rows, A.m_cols

	out = NewPrep(nrows, ncols, out)

	for i = 1, ncols * nrows do
		out[i] = A[i] * scale
	end

	return out
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam MatrixMN B
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN D
function M.Sub (A, B, out)
	local nrows, ncols = A.m_rows, A.m_cols

	assert(nrows == B.m_rows, "Mismatched rows")
	assert(ncols == B.m_cols, "Mismatched columns")
-- TODO: Zero-pad?
	out = NewPrep(nrows, ncols, out)

	for index = 1, nrows * ncols do
		out[index] = A[index] - B[index]
	end

	return out
end

--- DOCME
-- @tparam MatrixMN A
-- @tparam[opt] MatrixMN out
-- @treturn MatrixMN T
function M.Transpose (A, out)
	local nrows, ncols, index = A.m_rows, A.m_cols, 1

	out = NewPrep(ncols, nrows, out)

	for col = 1, ncols do
		local ci = col

		for _ = 1, nrows do
			out[index], index, ci = A[ci], index + 1, ci + ncols
		end
	end

	return out
end

--- DOCME
-- @uint nrows
-- @uint ncols
-- @treturn MatrixMN m
function M.Zero (nrows, ncols)
	return ZeroPrep(nrows, ncols)
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

	-- IsVector(), IsScalar()?
	-- Multiply(), TransposeMultiply()?
	-- Rank()?
	-- column, row length, length squared
	-- column, row dot products
	-- Plus(), Minus(), Times()...

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
		local arr, ncols, index = {}, self.m_cols

		for _ = from or 1, self.m_rows do
			index = index or Index(self, from or 1, col)

			arr[#arr + 1], index = self[index], index + ncols
		end

		return arr
	end

	--- DOCME
	-- @treturn uint NCOLS
	function MatrixMethods:GetColumnCount ()
		return self.m_cols
	end

	--- DOCME
	-- @treturn uint R
	-- @treturn uint C
	function MatrixMethods:GetDims ()
		return self.m_rows, self.m_cols
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

	--- DOCME
	-- @treturn uint NROWS
	function MatrixMethods:GetRowCount ()
		return self.m_rows
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
	-- @uint row
	-- @uint col
	-- @number value
	function MatrixMethods:Set (row, col, value)
		self[Index(self, row, col)] = value
	end

	--- DOCME
	-- @uint row
	-- @uint col
	-- @number delta
	function MatrixMethods:Update (row, col, delta)
		local index = Index(self, row, col)

		self[index] = self[index] + delta
	end
end

-- Cache module members.
_Columns_From_ = M.Columns_From
_New_ = M.New

-- Export the module.
return M