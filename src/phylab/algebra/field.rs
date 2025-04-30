//! Fields
use std::marker::PhantomData;

use super::ops::Binary;
use crate::utils::bool::True;

/// A field
pub trait Field<
    Add: Binary<Assoc = True, Commut = True> + ?Sized,
    Mul: Binary<Assoc = True, Commut = True> + ?Sized,
>: Sized
{
    /// The addition operation
    fn add(&self, rhs: &Self) -> Self;
    /// The additive inverse
    fn neg(&self) -> Self;
    /// The multiplication operation
    fn mul(&self, rhs: &Self) -> Self;
    /// The multiplicative inverse (should panic of `self` divides zero)
    fn inv(&self) -> Self {
        self.checked_inv()
            .expect("Error: zero divisors have no multiplicative inverse!")
    }
    /// The multiplicative inverse (should return `None` if `self` divides zero)
    fn checked_inv(&self) -> Option<Self>;
    /// The neutral element wrt/ addition
    fn zero() -> Self;
    /// The neutral element wrt/ multiplication
    fn one() -> Self;

    /// My `n`-th power (wrt/ multiplication)
    fn pow(&self, n: isize) {
        self.as_mul_group_ref().pow(n)
    }

    fn as_add_group(self) -> AsAdd<Add, Mul, Self> {
        AsAdd::wrap(self)
    }
    fn as_mul_group(self) -> AsMul<Add, Mul, Self> {
        AsAdd::wrap(self)
    }
    fn as_add_group_ref(&self) -> &AsAdd<Add, Mul, Self> {
        AsAdd::wrap_ref(self)
    }
    fn as_mul_group_ref(&self) -> &AsMul<Add, Mul, Self> {
        AsMul::wrap_ref(self)
    }
}

/// Helper for operating with a field in additive and multiplicative notation.
#[repr(transparent)]
pub struct AsAddMul<
    Add: Binary<Assoc = True, Commut = True> + ?Sized,
    Mul: Binary<Assoc = True, Commut = True> + ?Sized,
    F: Field<Add, Mul>,
>(F, PhantomData<Add>, PhantomData<Mul>);
impl<
    Add: Binary<Assoc = True, Commut = True> + ?Sized,
    Mul: Binary<Assoc = True, Commut = True> + ?Sized,
    F: Field<Add, Mul>,
> AsAddMul<Add, Mul, F>
{
    fn wrap(elem: F) -> Self {
        Self(elem, PhantomData, PhantomData)
    }
    fn unwrap(self) -> F {
        self.0
    }
    fn wrap_ref(elem: &F) -> &Self {
        // SAFETY: This is sound because `Self` is `#[repr(transparent)]` over `F`.
        unsafe { &*(elem as *const F as *const Self) }
    }
    fn unwrap_ref(&self) -> &F {
        &self.0
    }
    /// The neutral element wrt/ addition
    #[inline(always)]
    fn zero() -> Self {
        Self::wrap(F::zero())
    }
    /// The neutral element wrt/ multiplication
    #[inline(always)]
    fn one() -> Self {
        Self::wrap(F::one())
    }
}

/// Helper for operating with a group in additive notation.
#[repr(transparent)]
pub struct AsAdd<
    Add: Binary<Assoc = True, Commut = True> + ?Sized,
    Mul: Binary<Assoc = True, Commut = True> + ?Sized,
    F: Field<Add, Mul>,
>(F, PhantomData<Add>, PhantomData<Mul>);
impl<
    Add: Binary<Assoc = True, Commut = True> + ?Sized,
    Mul: Binary<Assoc = True, Commut = True> + ?Sized,
    F: Field<Add, Mul>,
> AsAdd<Add, Mul, F>
{
    fn wrap(elem: F) -> Self {
        Self(elem, PhantomData, PhantomData)
    }
    fn unwrap(self) -> F {
        self.0
    }
    fn wrap_ref(elem: &F) -> &Self {
        // SAFETY: This is sound because `Self` is `#[repr(transparent)]` over `G`.
        unsafe { &*(elem as *const F as *const Self) }
    }
    fn unwrap_ref(&self) -> &F {
        &self.0
    }
    /// The neutral element
    #[inline(always)]
    fn zero() -> Self {
        Self::wrap(F::id())
    }
    /// The inverse (you can also spell it `-self`)
    #[inline(always)]
    fn inv(&self) -> Self {
        Self::wrap(self.unwrap_ref().inv())
    }
}
impl<
    Add: Binary<Assoc = True, Commut = True> + ?Sized,
    Mul: Binary<Assoc = True, Commut = True> + ?Sized,
    G: Field<Add, Mul>,
> std::ops::Add for &AsAdd<Add, Mul, G>
{
    type Output = AsAdd<Add, Mul, G>;
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
pub struct AsMul<Op: Binary<Assoc = True> + ?Sized, F: Field<Op>>(G, PhantomData<Op>);
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
