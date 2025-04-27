//! Group-like structures
use super::ops::Binary;
use crate::utils::bool::*;
use std::{
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
};

/// A set equipped with an operation that:
///  * is associative
///  * has a neutral element
///  * has an inverse (unary operation)
pub trait Group<Op: Binary<Assoc = True> + ?Sized> {
    fn op(&self, rhs: &Self) -> Self;
    /// The neutral element / identity (both right and left)
    fn id() -> Self;
    /// The inverse (both right and left)
    fn inv(&self) -> Self;
    /// The `n`-th power
    ///
    /// NOTE: This uses a recursive algorithm with O(log n) time complexity.
    ///   If your group allows for a more efficient algorithm, you can implement
    ///   this function yourself for better performance.
    fn pow(&self, n: isize) -> Self
    where
        Self: Sized + Clone,
    {
        fn pow<Op: Binary<Assoc = True> + ?Sized, G: Group<Op> + Clone + Sized>(
            x: &G,
            n: usize,
        ) -> G {
            if n == 1 {
                x.clone()
            } else if n % 2 == 0 {
                let y = pow(x, n >> 1);
                y.op(&y)
            } else {
                let y = pow(x, n >> 1);
                y.op(&y).op(x)
            }
        }
        if n > 0 {
            pow(self, n as usize)
        } else if n == 0 {
            Self::id()
        } else {
            pow(&self.inv(), -n as usize)
        }
    }
}

/// Helper for operating with a group in additive notation.
#[repr(transparent)]
pub struct AsAdd<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>>(G, PhantomData<Op>);
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> AsAdd<Op, G> {
    fn wrap(elem: G) -> Self {
        Self(elem, PhantomData)
    }
    fn unwrap(self) -> G {
        self.0
    }
    fn wrap_ref(elem: &G) -> &Self {
        // SAFETY: This is sound because `Self` is `#[repr(transparent)]` over `G`.
        unsafe { &*(elem as *const G as *const Self) }
    }
    fn unwrap_ref(&self) -> &G {
        &self.0
    }
    /// The neutral element
    #[inline(always)]
    fn zero() -> Self {
        Self::wrap(G::id())
    }
    /// The inverse (you can also spell it `-self`)
    #[inline(always)]
    fn inv(&self) -> Self {
        Self::wrap(self.unwrap_ref().inv())
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Add for &AsAdd<Op, G> {
    type Output = AsAdd<Op, G>;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        AsAdd::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Add<&Self> for AsAdd<Op, G> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &Self) -> Self::Output {
        AsAdd::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Add<AsAdd<Op, G>> for &AsAdd<Op, G> {
    type Output = AsAdd<Op, G>;
    #[inline(always)]
    fn add(self, rhs: AsAdd<Op, G>) -> Self::Output {
        AsAdd::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Add for AsAdd<Op, G> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        AsAdd::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Neg for AsAdd<Op, G> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.inv()
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Neg for &AsAdd<Op, G> {
    type Output = AsAdd<Op, G>;
    fn neg(self) -> Self::Output {
        self.inv()
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Sub for &AsAdd<Op, G> {
    type Output = AsAdd<Op, G>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Sub<AsAdd<Op, G>> for &AsAdd<Op, G> {
    type Output = AsAdd<Op, G>;

    fn sub(self, rhs: AsAdd<Op, G>) -> Self::Output {
        self + -rhs
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Sub<&Self> for AsAdd<Op, G> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self + -rhs
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Sub for AsAdd<Op, G> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op> + Sized + Clone> Mul<isize> for AsAdd<Op, G> {
    type Output = Self;

    fn mul(self, rhs: isize) -> Self::Output {
        Self::wrap(self.unwrap_ref().pow(rhs))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op> + Sized + Clone> Mul<isize> for &AsAdd<Op, G> {
    type Output = AsAdd<Op, G>;

    fn mul(self, rhs: isize) -> Self::Output {
        AsAdd::wrap(self.unwrap_ref().pow(rhs))
    }
}

/// Helper for operating with a group in multiplicative notation.
#[repr(transparent)]
pub struct AsMul<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>>(G, PhantomData<Op>);
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> AsMul<Op, G> {
    fn wrap(elem: G) -> Self {
        Self(elem, PhantomData)
    }
    fn unwrap(self) -> G {
        self.0
    }
    fn wrap_ref(elem: &G) -> &Self {
        // SAFETY: This is sound because `Self` is `#[repr(transparent)]` over `G`.
        unsafe { &*(elem as *const G as *const Self) }
    }
    fn unwrap_ref(&self) -> &G {
        &self.0
    }
    /// The neutral element
    #[inline(always)]
    fn one() -> Self {
        Self::wrap(G::id())
    }
    /// The inverse
    #[inline(always)]
    fn inv(&self) -> Self {
        Self::wrap(self.unwrap_ref().inv())
    }
    /// The `n`-th power
    ///
    /// NOTE: This uses a recursive algorithm with O(log n) time complexity.
    fn pow(&self, n: isize) -> Self
    where
        G: Sized + Clone,
    {
        Self::wrap(self.0.pow(n))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Mul for &AsMul<Op, G> {
    type Output = AsMul<Op, G>;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        AsMul::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Mul<&Self> for AsMul<Op, G> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self::Output {
        AsMul::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Mul<AsMul<Op, G>> for &AsMul<Op, G> {
    type Output = AsMul<Op, G>;
    #[inline(always)]
    fn mul(self, rhs: AsMul<Op, G>) -> Self::Output {
        AsMul::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
impl<Op: Binary<Assoc = True> + ?Sized, G: Group<Op>> Mul for AsMul<Op, G> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        AsMul::wrap(self.0.op(rhs.unwrap_ref()))
    }
}
