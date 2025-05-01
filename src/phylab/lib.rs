mod dual;
pub use dual::*;
pub mod tensor2;
pub mod tensor_traits;

use pyo3::prelude::*;

#[allow(non_camel_case_types)]
#[pymodule]
#[pyo3(name = "_lib")]
#[pyo3(module = "rberga06.phylab")]
mod py {
    pub use crate::dual::pydual::*;
}
