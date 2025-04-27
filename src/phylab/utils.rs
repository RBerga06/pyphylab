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
