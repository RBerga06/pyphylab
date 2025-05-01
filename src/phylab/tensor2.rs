//! Tensors (second try).
use crate::tensor_traits::{Num, Real, ScalarShape, Shape, Tensor as _Tensor, TensorLike};
use rayon::{iter::repeatn, prelude::*};
use std::{fmt::Debug, fmt::Write};

/* --- Shapes --- */
impl ScalarShape for Vec<usize> {
    fn eq(&self, rhs: &Self) -> bool {
        self.len() == rhs.len() && self.par_iter().zip(rhs).all(|(x, y)| *x == *y)
    }
    fn display(&self) -> String {
        let mut s = format!("(");
        self.iter().for_each(|x| write!(s, "{x},").unwrap());
        write!(s, ")").unwrap();
        s
    }
}

/* --- Scalars --- */
// TODO: Implement `pow` and `ipow`
macro_rules! _ImplScalarTraitsForBuiltins {
    ($($T:ident $(num(zero = $Zero:expr, one = $One:expr $(, real(e = $E:expr, pi = $Pi:expr))?))?);*) => {$(
        impl TensorLike for $T {
            type Shape = ();
            fn shape(&self) -> &Self::Shape { &() }

            type Bool = bool;
            fn eq(&self, rhs: &Self) -> Self::Bool { std::cmp::PartialEq::eq(self, rhs) }
        }

        $(
            impl Num for $T {
                type Int = isize;
                fn neg(&self) -> Self { -self }
                fn add(&self, rhs: &Self) -> Self { self + rhs }
                fn sub(&self, rhs: &Self) -> Self { self - rhs }
                fn mul(&self, rhs: &Self) -> Self { self * rhs }
                fn div(&self, rhs: &Self) -> Self { self / rhs }
                fn inv(&self) -> Self { $One / self }
                fn ipow(&self, _rhs: &Self::Int) -> Self { todo!() }
                fn zero(_: &Self::Shape) -> Self { $Zero }
                fn one(_: &Self::Shape) -> Self { $One }
            }

            $(
                impl Real for $T {
                    /// Natual exponential
                    fn exp(&self) -> Self { $T::exp(*self) }
                    fn ln(&self) -> Self { $T::ln(*self) }
                    fn pow(&self, _rhs: &Self) -> Self { todo!() }
                    fn sin(&self) -> Self { $T::sin(*self) }
                    fn cos(&self) -> Self  { $T::cos(*self) }
                    fn tan(&self) -> Self  { $T::tan(*self) }
                    fn asin(&self) -> Self { $T::asin(*self) }
                    fn acos(&self) -> Self  { $T::acos(*self) }
                    fn atan(&self) -> Self  { $T::atan(*self) }
                    fn sinh(&self) -> Self { $T::sinh(*self) }
                    fn cosh(&self) -> Self  { $T::cosh(*self) }
                    fn tanh(&self) -> Self  { $T::tanh(*self) }
                    fn atan2(&self, rhs: &Self) -> Self { $T::atan2(*self, *rhs) }
                    fn e(_: &Self::Shape) -> Self { $E }
                    fn pi(_: &Self::Shape) -> Self { $Pi }
                }
            )?
        )?
    )*};
}
_ImplScalarTraitsForBuiltins! {
    bool;
    i8 num(zero = 0, one = 1);
    i16 num(zero = 0, one = 1);
    i32 num(zero = 0, one = 1);
    i64 num(zero = 0, one = 1);
    i128 num(zero = 0, one = 1);
    isize num(zero = 0, one = 1);
    f32 num(zero = 0.0, one = 1.0, real(e = std::f32::consts::E, pi = std::f32::consts::PI));
    f64 num(zero = 0.0, one = 1.0, real(e = std::f64::consts::E, pi = std::f64::consts::PI))
}

/// A n-dimensional vector that forwards all operations to its elements.
#[derive(Debug, Clone)]
pub struct Tensor<T: TensorLike> {
    data: Vec<T>,
    shape: <Self as TensorLike>::Shape,
}
impl<T: TensorLike> Tensor<T> {
    fn _ensure_shape_matches(&self, rhs: &Self) {
        if !self.shape.eq(&rhs.shape) {
            panic!(
                "TypeError: This operation requires shapes to match:\n  got `{}` and `{}`",
                self.shape.display(),
                rhs.shape.display()
            )
        }
    }
    /// Element-wise, shape-preserving `.map(...)` operation
    fn map<U: TensorLike<Shape = T::Shape>, F: Sync + Fn(&T) -> U>(&self, f: F) -> Tensor<U> {
        let data = self.data.par_iter().map(|x| f(x)).collect();
        let shape = self.shape.clone();
        Tensor { data, shape }
    }
    /// Element-wise `.zip(rhs).map(...)` operation
    fn map2<
        U: TensorLike<Shape = T::Shape>,
        V: TensorLike<Shape = T::Shape>,
        F: Sync + Fn(&T, &U) -> V,
    >(
        &self,
        rhs: &Tensor<U>,
        f: F,
    ) -> Tensor<V> {
        if !ScalarShape::eq(&self.shape, &rhs.shape) {
            panic!("Shapes do not match!")
        }
        let data = self
            .data
            .par_iter()
            .zip(&rhs.data)
            .map(|(x, y)| f(x, y))
            .collect();
        let shape = self.shape.clone();
        Tensor { data, shape }
    }
}
impl<T: TensorLike> TensorLike for Tensor<T> {
    type Shape = (Vec<usize>, T::Shape);
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    type Bool = Tensor<T::Bool>;
    fn eq(&self, rhs: &Self) -> Self::Bool {
        self._ensure_shape_matches(rhs);
        let shape = self.shape.clone();
        let data = self
            .data
            .par_iter()
            .zip(&rhs.data)
            .map(|(x, y)| x.eq(y))
            .collect();
        Tensor { shape, data }
    }
}
impl<T: TensorLike> _Tensor<T> for Tensor<T> {
    fn ones_like(x: T, shape: &<Self::Shape as Shape>::TensorShape) -> Self {
        let elt_shape = x.shape().clone();
        let data = repeatn(x, shape.par_iter().sum()).collect();
        let shape = (shape.clone(), elt_shape);
        Self { data, shape }
    }

    fn eq_0(&self, rhs: &T) -> Self::Bool {
        let data = self.data.par_iter().map(|x| x.eq(rhs)).collect();
        let shape = self.shape.clone();
        Tensor { data, shape }
    }
}
impl<T: Num> Num for Tensor<T> {
    type Int = Tensor<T::Int>;

    fn neg(&self) -> Self {
        self.map(|x| x.neg())
    }
    fn add(&self, rhs: &Self) -> Self {
        self.map2(rhs, |x, y| x.add(y))
    }
    fn sub(&self, rhs: &Self) -> Self {
        self.map2(rhs, |x, y| x.sub(y))
    }

    fn inv(&self) -> Self {
        self.map(|x| x.inv())
    }
    fn mul(&self, rhs: &Self) -> Self {
        self.map2(rhs, |x, y| x.mul(y))
    }
    fn div(&self, rhs: &Self) -> Self {
        self.map2(rhs, |x, y| x.div(y))
    }

    fn ipow(&self, rhs: &Self::Int) -> Self {
        self.map2(rhs, |x, y| x.ipow(y))
    }

    fn zero(shape: &Self::Shape) -> Self {
        Self::ones_like(T::zero(shape.element()), shape.tensor())
    }
    fn one(shape: &Self::Shape) -> Self {
        Self::ones_like(T::one(shape.element()), shape.tensor())
    }
}
macro_rules! _ImplTensorUnaryOps {
    ($($f:ident),*) => { $(
        fn $f(&self) -> Self {
            self.map(|x| x.$f())
        }
    )*};
}
impl<T: Real> Real for Tensor<T> {
    _ImplTensorUnaryOps!(exp, ln, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan);
    fn atan2(&self, rhs: &Self) -> Self {
        self.map2(rhs, |x, y| x.atan2(y))
    }
    fn e(shape: &Self::Shape) -> Self {
        Self::ones_like(T::e(shape.element()), shape.tensor())
    }
    fn pi(shape: &Self::Shape) -> Self {
        Self::ones_like(T::pi(shape.element()), shape.tensor())
    }
}

/*

/// A wrapper around `Tensor` that treats the first dimension of the array as a vector.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Vector<T: Shaped>(Tensor<T>);
impl<T: Shaped> ScalarLike for Vector<T> {
    type Scalar = T;
    type Shape = (usize, T::Shape);
    fn shape(&self) -> &Self::Shape {
        self.0.shape()[0]
    }
}
impl<T: Shaped> VecOp for Vector<T> {
    fn dot(&self, rhs: &Self) -> Self::Scalar {
        todo!()
    }
}
impl<T: Shaped> Into<Tensor<T>> for Vector<T> {
    fn into(self) -> Tensor<T> {
        self.0
    }
}

/* --- Python wrappers --- */

PyVariants! {Tensor<_> => PyTensor {
    bool => PyTensorBool;
    i64 => PyTensorInt;
    f64 => PyTensorFloat;
}}

PyVariants! {Vector<_> => PyVector {
    bool => PyVectorBool;
    i64 => PyVectorInt;
    f64 => PyVectorFloat;
}}
*/
