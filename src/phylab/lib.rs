mod algebra;
mod dual;
mod linalg;
mod tensor;
mod utils;
pub use algebra::*;
pub use dual::*;
pub use linalg::*;

use pyo3::prelude::*;

#[allow(non_camel_case_types)]
#[pymodule]
#[pyo3(name = "_lib")]
#[pyo3(module = "rberga06.phylab")]
mod py {
    pub use crate::dual::pydual::*;
}
