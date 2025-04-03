use pyo3::prelude::*;

struct Datum<T, E> {
    best: T,
    err: E,
}

#[pymodule]
mod py {
    use super::*;

    #[pyclass]
    struct Float32Datum;

    #[pyfunction]
    fn triple(x: usize) -> usize { x }
}
