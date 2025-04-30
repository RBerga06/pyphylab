/// Private module for defining sealed traits
mod sealed {
    pub trait Sealed {}
}

pub mod bool {
    /// Compile-time booleans
    pub trait Bool: super::sealed::Sealed {
        const VAL: bool;
    }
    /// Compile-time booleans
    pub trait IsTrue: super::sealed::Sealed {}
    /// Compile-time booleans
    pub trait IsFalse: super::sealed::Sealed {}
    /// Compile-time `True` value
    pub struct True;
    /// Compile-time `False` value
    pub struct False;
    impl super::sealed::Sealed for True {}
    impl super::sealed::Sealed for False {}
    impl IsTrue for True {}
    impl IsFalse for False {}
    impl Bool for True {
        const VAL: bool = true;
    }
    impl Bool for False {
        const VAL: bool = false;
    }
}

/// Helper traits for defining arithmetic operations in Rust
pub mod ops {
    #[macro_export]
    macro_rules! ImplRustOpAddRef {
        ($Struct:ident<$($T:ident$(: $($Bound:tt),*)?);*>: $body:expr) => {
            impl<$($T$(: $($Bound)+*)?),*> std::ops::Add for &$Struct<$($T),*> {
                type Output = $Struct<$($T),*>;
                fn add(self, rhs: &$Struct<$($T),*>) -> Self::Output { ($body)(self, rhs) }
            }
            impl<$($T$(: $($Bound)+*)?),*> std::ops::Add<&$Struct<$($T),*>> for $Struct<$($T),*> {
                type Output = $Struct<$($T),*>;
                fn add(self, rhs: &$Struct<$($T),*>) -> Self::Output { ($body)(&self, rhs) }
            }
            impl<$($T$(: $($Bound)+*)?),*> std::ops::Add<$Struct<$($T),*>> for &$Struct<$($T),*> {
                type Output = $Struct<$($T),*>;
                fn add(self, rhs: $Struct<$($T),*>) -> Self::Output { ($body)(self, &rhs) }
            }
            impl<$($T$(: $($Bound)+*)?),*> std::ops::Add for $Struct<$($T),*> {
                type Output = $Struct<$($T),*>;
                fn add(self, rhs: $Struct<$($T),*>) -> Self::Output { ($body)(&self, &rhs) }
            }
        };
    }

    pub trait AddSubNegRef {
        fn add(&self, other: &Self) -> Self;
        fn neg(&self) -> Self;
        fn sub(&self, other: &Self) -> Self {
            self + -other
        }
    }
}
