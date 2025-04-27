pub use pyo3::prelude::*;
use pyo3::{exceptions::PyTypeError, types::PyTuple};
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! ImplUnOp {
    ($($Trait:ident: fn $fn:ident($Rhs:ty) -> $Output:ty = $body:expr);*$(;)?) => {$(
        impl $Trait for $Rhs {
            type Output = $Output;
            fn $fn(self) -> Self::Output {
                ($body)(self)
            }
        }
    )*};
}
macro_rules! ImplBinOp {
    ($($Trait:ident: fn $fn:ident($Lhs:ty, $Rhs:ty) -> $Output:ty = $body:expr);*$(;)?) => {$(
        impl $Trait<$Rhs> for $Lhs {
            type Output = $Output;
            fn $fn(self, rhs: $Rhs) -> Self::Output {
                ($body)(self, rhs)
            }
        }
    )*};
}

macro_rules! PyUnion {
    ($(#[$($attr:tt)*])* $name:ident: $($variant:ident<$T:ty>),*) => {
        $(#[$($attr)*])*
        #[derive(Debug, Clone)]
        #[derive(FromPyObject, IntoPyObject)]
        enum $name {
            $(
                #[pyo3(transparent)]
                $variant($T),
            )*
        }

        $(
            impl From<$T> for $name {
                fn from(value: $T) -> Self {
                    Self::$variant(value)
                }
            }
        )*
    };
    ($(#[$($attr:tt)*])* $name:ident<'py>: $($variant:ident<$T:ty>),*) => {
        $(#[$($attr)*])*
        #[derive(Debug, Clone)]
        #[derive(FromPyObject, IntoPyObject)]
        enum $name<'py> {
            $(
                #[pyo3(transparent)]
                $variant($T),
            )*
        }

        $(
            impl<'py> From<$T> for $name<'py> {
                fn from(value: $T) -> Self {
                    Self::$variant(value)
                }
            }
        )*
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);
impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        let mut is_first = true;
        for x in &self.0 {
            if is_first {
                is_first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{x}")?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum SimdError {
    BadShape { lhs: Shape, rhs: Shape },
}
impl From<SimdError> for PyErr {
    fn from(value: SimdError) -> Self {
        match value {
            SimdError::BadShape { lhs, rhs } => PyTypeError::new_err(format!(
                "Mismatched shapes in binary operation: lhs is {lhs}, rhs is {rhs}",
            )),
        }
    }
}

/// The dot product of two vectors.
pub trait Dot<Rhs = Self> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

/// The matrix product of two matrices, a matrix and a vector or a vector and a matrix.
pub trait MatMul<Rhs = Self> {
    type Output;
    fn matmul(self, rhs: Rhs) -> Self::Output;
}

mod simd {
    use super::*;

    /// N-dimensional vector (dynamically-sized) where all operations happen element-wise
    ///
    /// Not actually SIMD, but at least [`rayon`]-parallelized.
    #[derive(Debug, Clone)]
    pub struct Simd<T> {
        // TODO: Implement this manually with raw pointers,
        //  because the `shape` already contains data size.
        shape: Shape,
        data: Vec<T>,
    }
    impl<T> Simd<T> {
        pub fn from_raw(shape: Shape, data: Vec<T>) -> Self {
            Self { shape, data }
        }
        pub fn raw_data(&self) -> &[T] {
            &self.data
        }
        pub fn shape(&self) -> &Shape {
            &self.shape
        }
        pub fn ensure_shape_matches(&self, rhs: &Self) -> Result<(), SimdError> {
            if self.shape != rhs.shape {
                Err(SimdError::BadShape {
                    lhs: self.shape.clone(),
                    rhs: rhs.shape.clone(),
                })
            } else {
                Ok(())
            }
        }
    }

    macro_rules! _ImplSimdUnOp {
        ($Op:ident: $op_fn:ident) => {
            impl<'a, T: Sync + Send> $Op for &'a Simd<T>
            where
                &'a T: $Op<Output = T>,
            {
                type Output = Simd<T>;

                fn $op_fn(self) -> Self::Output {
                    let shape = self.shape.clone();
                    let data = self.data.par_iter().map(|x| x.$op_fn()).collect();
                    Simd { shape, data }
                }
            }
        };
    }
    macro_rules! _ImplSimdBinOp {
        ($Op:ident: $op_fn:ident) => {
            impl<'a, T: Sync + Send> $Op for &'a Simd<T>
            where
                &'a T: $Op<Output = T>,
            {
                type Output = Simd<T>;

                fn $op_fn(self, rhs: Self) -> Self::Output {
                    self.ensure_shape_matches(rhs).unwrap();
                    let shape = self.shape.clone();
                    let data = self
                        .data
                        .par_iter()
                        .zip(&rhs.data)
                        .map(|(x, y)| x.$op_fn(y))
                        .collect();
                    Simd { shape, data }
                }
            }

            impl<'a, T: Sync + Send> $Op<&'a T> for &'a Simd<T>
            where
                &'a T: $Op<Output = T>,
            {
                type Output = Simd<T>;

                fn $op_fn(self, rhs: &'a T) -> Self::Output {
                    let shape = self.shape.clone();
                    let data = self.data.par_iter().map(|x| x.$op_fn(rhs)).collect();
                    Simd { shape, data }
                }
            }
        };
    }

    _ImplSimdUnOp!(Neg: neg);
    _ImplSimdBinOp!(Add: add);
    _ImplSimdBinOp!(Sub: sub);
    _ImplSimdBinOp!(Mul: mul);
    _ImplSimdBinOp!(Div: div);
    _ImplSimdBinOp!(Dot: dot);
    _ImplSimdBinOp!(MatMul: matmul);
}
pub use simd::Simd;

/// Python-like bool -> i64 coercion
trait IntoI64 {
    fn into_i64(self) -> i64;
}
impl IntoI64 for i64 {
    fn into_i64(self) -> i64 {
        self
    }
}
impl IntoI64 for bool {
    fn into_i64(self) -> i64 {
        self as i64
    }
}

/// Python-like bool -> i64 -> f64 coercion
trait IntoF64 {
    fn into_f64(self) -> f64;
}
impl IntoF64 for bool {
    fn into_f64(self) -> f64 {
        self as i64 as f64
    }
}
impl IntoF64 for i64 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

/// Python-like coercing addition
trait CoerceAdd {
    type Output;
    fn coerce_add(self) -> Self::Output;
}
ImplUnOp! {
    CoerceAdd: fn coerce_add((bool, bool)) -> i64 = { |(x, y)| x as i64 + y as i64 };
    CoerceAdd: fn coerce_add((bool, i64)) -> i64 = { |(x, y)| x as i64 + y };
    CoerceAdd: fn coerce_add((bool, f64)) -> f64 = { |(x, y)| x as i64 as f64 + y };
    CoerceAdd: fn coerce_add((i64, bool)) -> i64 = { |(x, y)| x + y as i64 };
    CoerceAdd: fn coerce_add((i64, i64)) -> i64 = { |(x, y)| x + y };
    CoerceAdd: fn coerce_add((i64, f64)) -> f64 = { |(x, y)| x as f64 + y };
    CoerceAdd: fn coerce_add((f64, bool)) -> f64 = { |(x, y)| x + y as i64 as f64 };
    CoerceAdd: fn coerce_add((f64, i64)) -> f64 = { |(x, y)| x + y as f64 };
    CoerceAdd: fn coerce_add((f64, f64)) -> f64 = { |(x, y)| x + y };
}

/// Python-like coercing multiplication
trait CoerceMul {
    type Output;
    fn coerce_mul(self) -> Self::Output;
}

macro_rules! PySimdImplPyOpScalar {
    ($pyclass:ident::$pyfn:ident { $body:expr }) => {
        #[pymethods]
        impl $pyclass {
            fn $pyfn<'py>(&self, rhs: PySimdLikeRef<'py>) -> PySimdVariant {
                match rhs {
                    PySimdLikeRef::Bool(b) => ($body)(self, b).into(),
                    PySimdLikeRef::Int64(b) => ($body)(self, b).into(),
                    PySimdLikeRef::Float(b) => ($body)(self, b).into(),
                    PySimdLikeRef::SimdBool(b) => ($body)(self, b.get()).into(),
                    PySimdLikeRef::SimdInt64(b) => ($body)(self, b.get()).into(),
                    PySimdLikeRef::SimdFloat(b) => ($body)(self, b.get()).into(),
                }
            }
        }
    };
}
macro_rules! PySimd {
    ($name:ident<$T:ty>) => {
        #[pyclass(frozen)]
        #[derive(Debug, Clone)]
        struct $name(Simd<$T>);
        #[pymethods]
        impl $name {
            #[getter]
            fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                PyTuple::new(py, self.0.shape().0.clone())
            }
        }
        PySimdImplPyOpScalar!($name::__add__ { |x, y| x + y });
        PySimdImplPyOpScalar!($name::__sub__ { |x, y| x - y });
        PySimdImplPyOpScalar!($name::__mul__ { |x, y| x * y });
        PySimdImplPyOpScalar!($name::__div__ { |x, y| x / y });
        PySimdImplPyOpScalar!($name::__radd__ { |x, y| x + y });
        PySimdImplPyOpScalar!($name::__rsub__ { |x, y| y - x });
        PySimdImplPyOpScalar!($name::__rmul__ { |x, y| x * y });
        PySimdImplPyOpScalar!($name::__rdiv__ { |x, y| y / x });
    };
}
macro_rules! _PySimdImplOpScalar {
    ($OpTrait:ident($op_fn:ident) for ($MySelf:ident), ($($ScalarRhs:ident | $SimdRhs:ident),*) -> $Output:ident { $func:expr }) => {
        $(
        impl<'a> $OpTrait<$ScalarRhs> for &'a $MySelf {
            type Output = $Output;
            fn $op_fn(self, rhs: $ScalarRhs) -> Self::Output {
                let shape = self.0.shape().clone();
                let data = self
                    .0
                    .raw_data()
                    .par_iter()
                    .map(|x| $func(*x, rhs))
                    .collect();
                $Output(Simd::from_raw(shape, data))
            }
        }

        impl<'a> $OpTrait<&'a $MySelf> for $ScalarRhs {
            type Output = $Output;
            fn $op_fn(self, rhs: &'a $MySelf) -> Self::Output {
                let shape = rhs.0.shape().clone();
                let data = rhs
                    .0
                    .raw_data()
                    .par_iter()
                    .map(|x| $func(self, *x))
                    .collect();
                $Output(Simd::from_raw(shape, data))
            }
        }

        impl<'a> $OpTrait<&'a $SimdRhs> for &'a $MySelf {
            type Output = $Output;
            fn $op_fn(self, rhs: &'a $SimdRhs) -> Self::Output {
                let shape = self.0.shape().clone();
                let data = self
                    .0
                    .raw_data()
                    .par_iter()
                    .zip(rhs.0.raw_data())
                    .map(|(x, y)| $func(*x, *y))
                    .collect();
                $Output(Simd::from_raw(shape, data))
            }
        }
    )*
    };
}
macro_rules! PySimdImplOpScalar {
    ($OpTrait:ident($op_fn:ident) for ($($MySelf:ident),*), $rhs:tt -> $Output:ident { $func:expr }) => {
        $(_PySimdImplOpScalar!($OpTrait($op_fn) for ($MySelf), $rhs -> $Output { $func });)*
    };
}

/// Define everything in one go
macro_rules! PySimdAll {
    (
        $(
            #[variant($Variant:ident)]
            $PySimdName:ident<$T:ty, ref<'py> = $TRef:ty>
        ),*$(,)?
    ) => {
        $(
            PySimd!($PySimdName<$T>);
        )*

        PyUnion!(PySimdVariant: $($Variant<$PySimdName>),*);
        PyUnion!(PySimdVariantRef<'py>: $($Variant<Bound<'py, $PySimdName>>),*);
        PyUnion!(PySimdLikeRef<'py>: $($Variant<$TRef>),*);
    }
}

PySimdAll! {
    #[variant(Bool)]
    PySimdBool<bool, ref<'py> = bool>,
    #[variant(Int64)]
    PySimdInt64<i64, ref<'py> = i64>,
    #[variant(Float)]
    PySimdFloat<f64, ref<'py> = f64>,
    #[variant(Simd)]
    PySimdSimd<PySimdVariant, ref<'py> = PySimdVariantRef<'py>>,
}
mod _py_simd_ops {
    use super::*;

    PySimdImplOpScalar!(Add(add) for (PySimdBool, PySimdInt64), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdInt64 { |x, y| x as i64 + y as i64 });
    PySimdImplOpScalar!(Sub(sub) for (PySimdBool, PySimdInt64), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdInt64 { |x, y| x as i64 - y as i64 });
    PySimdImplOpScalar!(Mul(mul) for (PySimdBool, PySimdInt64), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdInt64 { |x, y| x as i64 * y as i64 });
    PySimdImplOpScalar!(Div(div) for (PySimdBool, PySimdInt64), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdFloat { |x, y| x as i64 as f64 / y as i64 as f64 });

    PySimdImplOpScalar!(Add(add) for (PySimdBool, PySimdInt64), (f64 | PySimdFloat) -> PySimdFloat { |x, y| IntoF64::into_f64(x) + IntoF64::into_f64(y) });
    PySimdImplOpScalar!(Sub(sub) for (PySimdBool, PySimdInt64), (f64 | PySimdFloat) -> PySimdFloat { |x, y| IntoF64::into_f64(x) - IntoF64::into_f64(y) });
    PySimdImplOpScalar!(Mul(mul) for (PySimdBool, PySimdInt64), (f64 | PySimdFloat) -> PySimdFloat { |x, y| IntoF64::into_f64(x) * IntoF64::into_f64(y) });
    PySimdImplOpScalar!(Div(div) for (PySimdBool, PySimdInt64), (f64 | PySimdFloat) -> PySimdFloat { |x, y| IntoF64::into_f64(x) / IntoF64::into_f64(y) });

    PySimdImplOpScalar!(Add(add) for (PySimdFloat), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdFloat { |x, y| IntoF64::into_f64(x) + IntoF64::into_f64(y) });
    PySimdImplOpScalar!(Sub(sub) for (PySimdFloat), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdFloat { |x, y| IntoF64::into_f64(x) - IntoF64::into_f64(y) });
    PySimdImplOpScalar!(Mul(mul) for (PySimdFloat), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdFloat { |x, y| IntoF64::into_f64(x) * IntoF64::into_f64(y) });
    PySimdImplOpScalar!(Div(div) for (PySimdFloat), (bool | PySimdBool, i64 | PySimdInt64) -> PySimdFloat { |x, y| IntoF64::into_f64(x) / IntoF64::into_f64(y) });

    PySimdImplOpScalar!(Add(add) for (PySimdFloat), (f64 | PySimdFloat) -> PySimdFloat { |x, y| x + y });
    PySimdImplOpScalar!(Sub(sub) for (PySimdFloat), (f64 | PySimdFloat) -> PySimdFloat { |x, y| x - y });
    PySimdImplOpScalar!(Mul(mul) for (PySimdFloat), (f64 | PySimdFloat) -> PySimdFloat { |x, y| x * y });
    PySimdImplOpScalar!(Div(div) for (PySimdFloat), (f64 | PySimdFloat) -> PySimdFloat { |x, y| x / y });
}

fn foo<'py>(x: PySimdLikeRef<'py>) {
    match x {
        PySimdLikeRef::Bool(x) => {}
        PySimdLikeRef::Simd(x) => {}
    }
}
