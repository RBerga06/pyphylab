use super::ops::Unary;

/// A set with a unary operation defined on it.
///
/// Note: the operation is not necessarily closed on `Self`.
pub trait UnarySet<Op: Unary + ?Sized> {
    type Output;
    fn op(&self) -> Self::Output;
}

/// A [`UnarySet`] where the unary operation is closed.
///
/// (essentially an alias for [`UnarySet<Op, Output = Self>`])
pub trait UnaryClosedSet<Op: Unary + ?Sized>: UnarySet<Op, Output = Self> {}
impl<Op: Unary + ?Sized, S: UnarySet<Op, Output = Self>> UnaryClosedSet<Op> for S {}

/// A [`UnarySet`] where the unary operation accepts an inverse.
///
/// (essentially an alias for `UnarySet<Op> where Self::Output: UnarySet<Op::Inv, Output = Self>`)
pub trait UnaryInvSet<Op: Unary + ?Sized>: UnarySet<Op>
where
    Self::Output: UnarySet<Op::Inv, Output = Self>,
{
}
impl<Op: Unary + ?Sized, S: UnarySet<Op>> UnaryInvSet<Op> for S where
    Self::Output: UnarySet<Op::Inv, Output = Self>
{
}
