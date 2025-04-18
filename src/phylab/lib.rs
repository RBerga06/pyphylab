use std::fmt::Display;

use num_traits::{Float, Num};
use numpy::ndarray::{Array1, ScalarOperand};
use pyo3::prelude::*;

/// Either a scalar or an array
#[derive(Debug, Clone)]
pub enum Value<T: Num> {
    Scalar(T),
    Array(Array1<T>),
}
impl<T: Num> From<T> for Value<T> {
    fn from(value: T) -> Self {
        Self::Scalar(value)
    }
}
impl<T: Num> From<Array1<T>> for Value<T> {
    fn from(value: Array1<T>) -> Self {
        Self::Array(value)
    }
}

#[derive(Debug, Clone)]
pub struct Scalar<T: Num + Copy> {
    pub best: T,
    pub delta: T,
}
impl<T: Num + Copy> Scalar<T> {
    pub fn from_best(best: T) -> Self {
        Self {
            best,
            delta: T::zero().into(),
        }
    }
    pub fn from_delta(best: T, delta: T) -> Self {
        Self { best, delta }
    }
    pub fn from_delta_rel(best: T, delta_rel: T) -> Self {
        Self {
            best,
            delta: best * delta_rel,
        }
    }
    pub fn delta_rel(&self) -> Option<T> {
        if self.best == T::zero() {
            None
        } else {
            Some(self.delta / self.best)
        }
    }
}
impl<T: Num + Copy> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Self::from_best(value)
    }
}
impl<T: Num + Copy + Display> Display for Scalar<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: Improve this
        write!(f, "{} ± {}", self.best, self.delta)
    }
}

#[derive(Debug, Clone)]
pub struct Vector<T: Float + Copy + ScalarOperand> {
    pub best: Array1<T>,
    pub delta: Value<T>,
}
impl<T: Float + Copy + ScalarOperand> Vector<T> {
    pub fn from_best(best: Array1<T>) -> Self {
        Self {
            best,
            delta: T::zero().into(),
        }
    }
    pub fn from_delta(best: Array1<T>, delta: T) -> Self {
        Self {
            best,
            delta: delta.into(),
        }
    }
    pub fn from_deltas(best: Array1<T>, delta: Array1<T>) -> Self {
        Self {
            best,
            delta: delta.into(),
        }
    }
    pub fn from_delta_rel(best: Array1<T>, delta_rel: T) -> Self {
        let delta: Array1<T> = &best * delta_rel;
        Self {
            best,
            delta: delta.into(),
        }
    }
    pub fn from_deltas_rel(best: Array1<T>, delta_rel: Array1<T>) -> Self {
        let delta: Array1<T> = &best * delta_rel;
        Self {
            best,
            delta: delta.into(),
        }
    }
    pub fn deltas_rel(&self) -> Array1<T> {
        match &self.delta {
            Value::Scalar(delta) => self.best.recip() * *delta,
            Value::Array(delta) => self.best.recip() * delta,
        }
    }
}

#[allow(non_camel_case_types)]
#[pymodule]
#[pyo3(name = "_lib")]
#[pyo3(module = "rberga06.phylab")]
mod py {
    /// Internal utilities
    mod _utils {
        use numpy::{AllowTypeChange, IntoPyArray, PyArrayLike1, PyArrayMethods, ndarray::Array1};
        use pyo3::prelude::*;

        /// A Python-friendly wrapper around [`Array1<f64>`]
        #[derive(Debug, Clone)]
        pub struct vf64(pub Array1<f64>);
        impl<'py> FromPyObject<'py> for vf64 {
            fn extract_bound(ob: &pyo3::Bound<'py, PyAny>) -> Result<Self, PyErr> {
                let array: PyArrayLike1<'py, f64, AllowTypeChange> = ob.extract()?;
                Ok(vf64(array.to_owned_array()))
            }
        }
        impl<'py> IntoPyObject<'py> for vf64 {
            type Target = PyAny; // the Python type
            type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
            type Error = std::convert::Infallible;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                Ok(self.0.into_pyarray(py).into_any())
            }
        }
        impl From<Array1<f64>> for vf64 {
            fn from(value: Array1<f64>) -> Self {
                Self(value)
            }
        }
        impl Into<Array1<f64>> for vf64 {
            fn into(self) -> Array1<f64> {
                self.0
            }
        }

        /// An `f64` or an `Array<f64>`
        #[derive(Debug, Clone, FromPyObject, IntoPyObject)]
        pub enum f64Value {
            Scalar(f64),
            Array(vf64),
        }
    }

    use pyo3::exceptions::PyValueError;

    use super::*;

    #[pyclass]
    struct uf64(Scalar<f64>);
    #[pymethods]
    impl uf64 {
        #[new]
        #[pyo3(signature = (best = 0.0, /, delta = 0.0, *, delta_rel = None))]
        fn new(best: f64, delta: f64, delta_rel: Option<f64>) -> Self {
            if let Some(delta_rel) = delta_rel {
                Self(Scalar::from_delta_rel(best, delta_rel))
            } else {
                Self(Scalar::from_delta(best, delta))
            }
        }

        #[getter]
        fn get_best(&self) -> f64 {
            self.0.best
        }
        #[setter]
        fn set_best(&mut self, best: f64) {
            self.0.best = best;
        }
        #[getter]
        fn get_delta(&self) -> f64 {
            self.0.delta
        }
        #[setter]
        fn set_delta(&mut self, delta: f64) {
            self.0.delta = delta;
        }
        #[getter]
        fn get_delta_rel(&self) -> PyResult<f64> {
            self.0.delta_rel().ok_or_else(|| PyValueError::new_err("Encountered division by zero when evaluating relative uncertainty for a `uf64`!"))
        }
        #[setter]
        fn set_delta_rel(&mut self, delta_rel: f64) {
            self.0.delta = self.0.best * delta_rel;
        }

        fn __repr__(&self) -> String {
            format!("uf64({}, {})", self.0.best, self.0.delta)
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
    }

    #[pyclass]
    struct uvf64(Vector<f64>);
    #[pymethods]
    impl uvf64 {
        #[new]
        #[pyo3(signature = (best, /, delta = _utils::f64Value::Scalar(0.0), *, delta_rel = None))]
        fn new(
            best: _utils::vf64,
            delta: _utils::f64Value,
            delta_rel: Option<_utils::f64Value>,
        ) -> Self {
            let best: Array1<f64> = best.into();
            if let Some(delta_rel) = delta_rel {
                Self(match delta_rel {
                    _utils::f64Value::Scalar(delta_rel) => Vector::from_delta_rel(best, delta_rel),
                    _utils::f64Value::Array(delta_rel) => {
                        Vector::from_deltas_rel(best, delta_rel.into())
                    }
                })
            } else {
                Self(match delta {
                    _utils::f64Value::Scalar(delta) => Vector::from_delta(best, delta),
                    _utils::f64Value::Array(delta) => Vector::from_deltas(best, delta.into()),
                })
            }
        }

        #[getter]
        fn get_best(&self) -> _utils::vf64 {
            self.0.best.clone().into()
        }
        #[setter]
        fn set_best(&mut self, best: _utils::vf64) {
            self.0.best = best.into();
        }
        #[getter]
        fn get_delta(&self) -> _utils::f64Value {
            match &self.0.delta {
                Value::Scalar(x) => _utils::f64Value::Scalar((*x).into()),
                Value::Array(x) => _utils::f64Value::Array(x.clone().into()),
            }
        }
        #[setter]
        fn set_delta(&mut self, delta: _utils::f64Value) {
            self.0.delta = match delta {
                _utils::f64Value::Scalar(x) => Value::Scalar(x.into()),
                _utils::f64Value::Array(x) => Value::Array(x.into()),
            };
        }
        #[getter]
        fn get_delta_rel(&self) -> _utils::vf64 {
            self.0.deltas_rel().into()
        }
        #[setter]
        fn set_delta_rel(&mut self, delta_rel: _utils::f64Value) {
            match delta_rel {
                _utils::f64Value::Scalar(delta_rel) => {
                    self.0.delta = Value::Array(&self.0.best * delta_rel)
                }
                _utils::f64Value::Array(delta_rel) => {
                    self.0.delta = Value::Array(&self.0.best * delta_rel.0)
                }
            }
        }

        fn __repr__(&self) -> String {
            // format!("uv64({}, {})", self.0.best, self.0.delta)
            todo!()
        }
        fn __str__(&self) -> String {
            // format!("{}", self.0)
            todo!()
        }
    }
}
