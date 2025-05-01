//! Tensor traits
//!
//! TODO: Implement vector/matrix typed shape support, while still allowing for dynamically shaped types.
use std::fmt::Debug;

/// A scalar shape
pub trait ScalarShape: Sync + Send + Sized + Clone + Debug {
    /// Compare two shapes: are they equal?
    fn eq(&self, rhs: &Self) -> bool;
    /// Present this shape in a nice format for error messages
    fn display(&self) -> String;
}
/// A tensor shape
pub trait Shape: ScalarShape {
    type TensorShape: ScalarShape;
    type ElementShape: ScalarShape;
    fn tensor(&self) -> &Self::TensorShape;
    fn element(&self) -> &Self::ElementShape;
}
impl<T: ScalarShape, E: ScalarShape> ScalarShape for (T, E) {
    fn eq(&self, rhs: &Self) -> bool {
        self.0.eq(&rhs.0) && self.1.eq(&rhs.1)
    }
    fn display(&self) -> String {
        format!("<{} of {}>", self.0.display(), self.1.display())
    }
}
impl<T: ScalarShape, E: ScalarShape> Shape for (T, E) {
    type TensorShape = T;
    type ElementShape = E;
    fn element(&self) -> &Self::ElementShape {
        &self.1
    }
    fn tensor(&self) -> &Self::TensorShape {
        &self.0
    }
}
impl ScalarShape for () {
    fn eq(&self, _: &Self) -> bool {
        true
    }
    fn display(&self) -> String {
        format!("()")
    }
}
impl Shape for () {
    type TensorShape = ();
    type ElementShape = ();
    fn element(&self) -> &Self::ElementShape {
        &()
    }
    fn tensor(&self) -> &Self::ElementShape {
        &()
    }
}

/// Anything with scalar operations, a shape and an element type
pub trait TensorLike: Sync + Send + Clone + Debug {
    /// My shape (should be `()` for scalars).
    type Shape: Shape;
    /// Get shape information
    fn shape(&self) -> &Self::Shape;

    /// The result of element-wise boolean operations
    type Bool: TensorLike<Shape = Self::Shape>;
    /// Compute `self == rhs` (element-wise).
    fn eq(&self, rhs: &Self) -> Self::Bool;
}
/// [`ScalarLike`] with numeric operations
pub trait Num: TensorLike {
    /// The result / argument of integer operations (typically `isize`)
    type Int: TensorLike<Shape = Self::Shape>;

    /// `-self`
    fn neg(&self) -> Self;
    /// `self + rhs`
    fn add(&self, rhs: &Self) -> Self;
    /// `self - rhs`
    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }
    /// `1/self`
    fn inv(&self) -> Self;
    /// `self * rhs`
    fn mul(&self, rhs: &Self) -> Self;
    /// `self / rhs`
    fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.inv())
    }
    /// `self ** n`
    fn ipow(&self, n: &Self::Int) -> Self;

    /// 0
    fn zero(shape: &Self::Shape) -> Self;
    /// 1
    fn one(shape: &Self::Shape) -> Self;
    /// 2
    fn two(shape: &Self::Shape) -> Self {
        Self::one(shape).add(&Self::one(shape))
    }
}
/// Real [`Num`]bers
pub trait Real: Num {
    /// Natual exponential
    fn exp(&self) -> Self;
    /// Natual logarithm
    fn ln(&self) -> Self;
    /// Any power
    fn pow(&self, rhs: &Self) -> Self {
        (self.ln().mul(rhs)).exp()
    }
    /// Sine
    fn sin(&self) -> Self;
    /// Cosine
    fn cos(&self) -> Self;
    /// Tangent
    fn tan(&self) -> Self {
        self.sin().div(&self.cos())
    }
    /// Hyperbolic sine
    fn sinh(&self) -> Self;
    /// Hyperbolic cosine
    fn cosh(&self) -> Self;
    /// Hyperbolic tangent
    fn tanh(&self) -> Self {
        self.sinh().div(&self.cosh())
    }
    /// Inverse sine
    fn asin(&self) -> Self;
    /// Inverse cosine
    fn acos(&self) -> Self;
    /// Inverse tangent
    fn atan(&self) -> Self;
    /// `atan2(x, y)`, which returns an angle in [-π/2, π/2)
    fn atan2(&self, rhs: &Self) -> Self;

    fn e(shape: &Self::Shape) -> Self;
    fn pi(shape: &Self::Shape) -> Self;
}

/// Anything with scalar or tensor (i.e. distributed scalar) operations
pub trait Tensor<X: TensorLike>: TensorLike {
    /// Construct a tensor with the specified shape, where all elements are `x`
    fn ones_like(x: X, shape: &<Self::Shape as Shape>::TensorShape) -> Self;

    /// Compute `x == rhs` for all elements `x`.
    fn eq_0(&self, rhs: &X) -> Self::Bool;
}
/// `Scalar` with numeric operations
pub trait NumTensor<X: Num>: Tensor<X> + Num {
    /// `x + rhs` for all `x`s
    fn add_0(&self, rhs: &X) -> Self;
    /// `x - rhs` for all `x`s
    fn sub_0(&self, rhs: &X) -> Self {
        self.add_0(&rhs.neg())
    }
    /// `x - rhs` for all `x`s
    fn rsub_0(&self, rhs: &X) -> Self {
        self.neg().add_0(&rhs)
    }
    /// `x * rhs` for all `x`s
    fn mul_0(&self, rhs: &X) -> Self;
    /// `x / rhs` for all `x`s
    fn div_0(&self, rhs: &X) -> Self {
        self.mul_0(&rhs.inv())
    }
    /// `lhs / x` for all `x`s
    fn rdiv_0(&self, rhs: &X) -> Self {
        self.inv().mul_0(&rhs)
    }
    /// `x ** n` for all `x`s
    fn ipow_0(&self, n: &X::Int) -> Self;
}
/// Real [`Tensor`]s
pub trait RealTensor<X: Real>: NumTensor<X> + Real {
    /// Any power
    fn pow_0(&self, rhs: &X) -> Self {
        (self.ln().mul_0(rhs)).exp()
    }
    /// `atan2(x, y)`, which returns an angle in [-π/2, π/2)
    fn atan2_0(&self, rhs: &X) -> Self;
}

/// Vector
pub trait Vector<X: TensorLike>: Tensor<X> {}
/// Numeric vector
pub trait NumVector<X: Num>: Vector<X> + Num {
    /// Dot product between two vectors
    fn dot(&self, rhs: &X) -> Self::Int;
}

/// Matrix
pub trait Matrix<X: TensorLike>: Tensor<X> {}
/// Numeric matrix
pub trait NumMatrix<X: Num>: Matrix<X> + Num {
    /// Matrix product: `self (Matrix) @ rhs (Matrix)`
    fn matmat(&self, rhs: &Self) -> Self;
    /// Matrix product: `self (Matrix) @ rhs (Vector)`
    fn matvec(&self, rhs: &Self) -> Self;
    /// Matrix product: `lhs (Vector) @ self (Matrix)`
    fn rvecmat(&self, lhs: &Self) -> Self;
}
