--- Quaternion utilities.

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
   
-- Standard library imports --
local acos = math.acos
local cos = math.cos
local exp = math.exp
local log = math.log
local pi = math.pi
local sin = math.sin
local sqrt = math.sqrt

-- Modules --
local robust = require("tektite_core.number.robust")

-- Cached module references --
local _Add_
local _Add_Scaled_
local _AngleBetween_
local _Conjugate_
local _Dot_
local _Exp_
local _FromAxisAngle_
local _Inverse_
local _Length_
local _Log_
local _Multiply_
local _Negate_
local _Normalize_
local _Scale_
local _Slerp_
local _SquadAuxQuats_
local _SquadQ2S2_ 

-- Exports --
local M = {}

--- DOCME
function M.Add (qout, q1, q2)
	qout.x = q1.x + q2.x
	qout.y = q1.y + q2.y
	qout.z = q1.z + q2.z
	qout.w = q1.w + q2.w

	return qout
end

--- DOCME
function M.Add_Scaled (qout, q1, q2, k)
	qout.x = q1.x + q2.x * k
	qout.y = q1.y + q2.y * k
	qout.z = q1.z + q2.z * k
	qout.w = q1.w + q2.w * k

	return qout
end

-- Forward references --
local AuxAngleBetween

do
	local A1, A2, TwoPi = {}, {}, 2 * pi

	--- DOCME
	function M.AngleBetween (q1, q2)
		local angle = 2 * AuxAngleBetween(_Normalize_(A1, q1), _Normalize_(A2, q2))

		return angle > pi and TwoPi - angle or angle
	end
end

--- DOCME
function M.Conjugate (qout, q)
	qout.x = -q.x
	qout.y = -q.y
	qout.z = -q.z
	qout.w = q.w

	return qout
end

do
	local Qi = {}

	--- DOCME
	function M.Difference (qout, q1, q2)
		return _Multiply_(qout, _Inverse_(Qi, q1), q2)
	end
end

--- DOCME
function M.Dot (q1, q2)
	return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
end

--- DOCME
-- q = [a, theta * v], len(v) = 1 -> [e^a*cos(theta), e^a*sin(theta) * v]
function M.Exp (qout, q)
	local qx, qy, qz, ew = q.x, q.y, q.z, exp(q.w)
	local vnorm = sqrt(qx^2 + qy^2 + qz^2)
	local coeff = ew * robust.SinOverX(vnorm)

	qout.w, qout.x, qout.y, qout.z = ew * cos(vnorm), coeff * qx, coeff * qy, coeff * qz

	return qout
end

--- DOCME
function M.FromAxisAngle (qout, angle, vx, vy, vz)
	angle = .5 * angle

	local coeff = sin(angle) / sqrt(vx^2 + vy^2 + vz^2)

	qout.w, qout.x, qout.y, qout.z = cos(angle), coeff * vx, coeff * vy, coeff * vz

	return qout
end

do
	local Order, Axis = {
		xyz = function(x, y, z) return x, y, z end,
		xzy = function(x, y, z) return x, z, y end,
		yxz = function(x, y, z) return y, x, z end,
		yzx = function(x, y, z) return y, z, x end,
		zxy = function(x, y, z) return z, x, y end,
		zyx = function(x, y, z) return z, y, x end
	}, {}

	--- DOCME
	function M.FromEulerAngles (qout, x, y, z, method)
		local order = Order[method] or Order.xyz

		x, y, z = order(x, y, z)

		if method == "yzx" or method == "zxy" then -- axes are swapped in these two cases
			order = Order[method == "yzx" and "zxy" or "yzx"]
		end

		_FromAxisAngle_(qout, z, order(0, 0, 1))
		_FromAxisAngle_(Axis, y, order(0, 1, 0))
		_Multiply_(qout, Axis, qout)
		_FromAxisAngle_(Axis, x, order(1, 0, 0))
		_Multiply_(qout, Axis, qout)

		return qout
	end
end

--- DOCME
function M.Inverse (qout, q)
	return _Normalize_(qout, _Conjugate_(qout, q))
end

--- DOCME
function M.Length (q)
	return sqrt(q.x^2 + q.y^2 + q.z^2 + q.w^2)
end

--- DOCME
-- q = [len(q) * cos(theta), len(q) * sin(theta) * v] -> [ln(len(q)), acos(w / len(q)) * v], len(v) = 1
function M.Log (qout, q)
	-- Adapted from:
	-- https://github.com/numpy/numpy-dtypes/blob/76da931005a088f9e5f75d8ea2d58428cad2a975/npytypes/quaternion/quaternion.c#L121
	local qx, qy, qz = q.x, q.y, q.z
	local sqr = qx^2 + qy^2 + qz^2
	local vnorm = sqrt(sqr)

	if vnorm > 1e-6 then
		local qw = q.w
		local mag = sqrt(sqr + qw^2)
		local coeff = acos(qw / mag) / vnorm

		qout.w, qout.x, qout.y, qout.z = log(mag), coeff * qx, coeff * qy, coeff * qz
	else
		qout.w, qout.x, qout.y, qout.z = 0, 0, 0, 0
	end

	return qout
end

--- DOCME
function M.Multiply (qout, q1, q2)
	local x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
	local x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w

	qout.x = w1 * x2 + w2 * x1 + y1 * z2 - y2 * z1
	qout.y = w1 * y2 + w2 * y1 + z1 * x2 - z2 * x1
	qout.z = w1 * z2 + w2 * z1 + x1 * y2 - x2 * y1
	qout.w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

	return qout
end

--- DOCME
function M.Negate (qout, q)
	qout.x = -q.x
	qout.y = -q.y
	qout.z = -q.z
	qout.w = -q.w

	return qout
end

--- DOCME
function M.Normalize (qout, q)
	return _Scale_(qout, q, 1 / _Length_(q))
end

--- DOCME
function M.Scale (qout, q, k)
	qout.x = q.x * k
	qout.y = q.y * k
	qout.z = q.z * k
	qout.w = q.w * k

	return qout
end

do
	local Qf, Qt = {}, {}, {}

	--- DOCME
	function M.Slerp (qout, q1, q2, t)
		_Normalize_(Qf, q1)
		_Normalize_(Qt, q2)

		local dot, k1, k2 = _Dot_(Qf, Qt)

		if dot < 0 then
			_Negate_(Qf, Qf)

			dot = -dot
		end

		if dot > .95 then
			k1, k2 = 1 - t, t
		else
			k1, k2 = robust.SlerpCoeffs(t, _AngleBetween_(Qf, Qt))
		end

		_Add_Scaled_(qout, _Scale_(Qf, Qf, k1), Qt, k2)

		return _Normalize_(qout, qout)
	end
end

do
	local Qa, Qb = {}, {}

	--- DOCME
	function M.SquadQ2S2 (qout, q1, q2, s1, s2, t)
		return _Slerp_(qout, _Slerp_(Qa, q1, q2, t), _Slerp_(Qb, s1, s2, t), 2 * t * (1 - t))
	end
end

do
	local Qi, Log1, Log2, Sum = {}, {}, {}, {}

	--- DOCME
	function M.SquadAuxQuats (qout, qprev, q, qnext)
		_Inverse_(Qi, q)
		_Log_(Log1, _Multiply_(Log1, Qi, qprev))
		_Log_(Log2, _Multiply_(Log2, Qi, qnext))
		_Scale_(Sum, _Add_(Sum, Log1, Log2), -.25)

		return _Multiply_(qout, _Exp_(Sum, Sum), q)
	end
end

do
	local S1, S2 = {}, {}

	--- DOCME
	function M.SquadQ4 (qout, q1, q2, q3, q4, t)
		return _SquadQ2S2_(qout, q2, q3, _SquadAuxQuats_(S1, q1, q2, q3), _SquadAuxQuats_(S2, q2, q3, q4), t)
	end
end

--- DOCME
function M.Sub (qout, q1, q2)
	qout.x = q1.x - q2.x
	qout.y = q1.y - q2.y
	qout.z = q1.z - q2.z
	qout.w = q1.w - q2.w

	return qout
end

--
AuxAngleBetween = robust.AngleBetween(M.Dot, M.Length, M.Sub)

-- Cache module members.
_Add_ = M.Add
_Add_Scaled_ = M.Add_Scaled
_AngleBetween_ = M.AngleBetween
_Conjugate_ = M.Conjugate
_Dot_ = M.Dot
_Exp_ = M.Exp
_FromAxisAngle_ = M.FromAxisAngle
_Inverse_ = M.Inverse
_Length_ = M.Length
_Log_ = M.Log
_Multiply_ = M.Multiply
_Negate_ = M.Negate
_Normalize_ = M.Normalize
_Scale_ = M.Scale
_Slerp_ = M.Slerp
_SquadAuxQuats_ = M.SquadAuxQuats
_SquadQ2S2_ = M.SquadQ2S2

-- Export the module.
return M