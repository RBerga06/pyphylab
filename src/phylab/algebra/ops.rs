//! Operations
use crate::utils::bool::{Bool, False};

use super::unary::UnarySet;
use std::{convert::Infallible, marker::PhantomData};

/// A unary operation
pub trait Unary {
    /// If an operation does not define an inverse, this should be `!`
    type Inv: Unary + ?Sized;

    /// The operation
    fn op<B, A: UnarySet<Self, Output = B>>(arg: &A) -> B {
        arg.op()
    }

    /// The inverse operation
    fn inv<A, B: UnarySet<Self::Inv, Output = A>>(arg: &B) -> A {
        Self::Inv::op(arg)
    }
}

/// Helper for creating an inverse for ordinary unary operations.
pub struct UnaryInverse<Op: Unary + ?Sized>(PhantomData<Op>);
impl<Op: Unary + ?Sized> Unary for UnaryInverse<Op> {
    type Inv = Op;
}

/// The unary operation that does not exist
// TODO: When we have `!` we should switch to it
impl Unary for Infallible {
    type Inv = Infallible;
}

/// The identity unary operation
pub struct Identity;
impl Unary for Identity {
    type Inv = Identity;
}
impl<T: Clone> UnarySet<Identity> for T {
    type Output = Self;
    fn op(&self) -> Self {
        self.clone()
    }
}

/// The unary operation that makes everything zero.
///
/// (mostly just an example of defining an unary operator with no inverse).
pub struct Nullify;
impl Unary for Nullify {
    type Inv = Infallible; // TODO: When we have `!` we should switch to it
}

/// A binary operation
pub trait Binary {
    type Assoc: Bool;
}

/// The binary operation that does not exist
// TODO: When we have `!` we should switch to it
impl Binary for Infallible {
    type Assoc = False;
}
