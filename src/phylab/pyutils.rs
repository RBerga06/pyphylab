//! Macros & other utils for implementing Python wrappers around Rust structures.

#[macro_export]
macro_rules! PyVariants {
    ($Struct:ident<_> => $PyStruct:ident { $($T:ty => $PyVariant:ident);*$(;)? }) => {
        $(
            #[pyo3::prelude::pyclass(frozen)]
            pub struct $PyVariant(pub $Struct<$T>);
            #[pyo3::prelude::pymethods]
            impl $PyVariant {
                #[new]
                fn new() -> Self { todo!() }
            }
        )*
    };
}
