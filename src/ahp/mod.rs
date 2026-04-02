use crate::{String, ToString, Vec};
use ark_ff::{Field, PrimeField};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_poly_commit::{LCTerm, LinearCombination};
use ark_relations::r1cs::SynthesisError;
use ark_std::{borrow::Borrow, cfg_iter_mut, format, marker::PhantomData, vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(crate) mod constraint_systems;
/// Describes data structures and the algorithms used by the AHP indexer.
pub mod indexer;
/// Describes data structures and the algorithms used by the AHP prover.
pub mod prover;
/// Describes data structures and the algorithms used by the AHP verifier.
pub mod verifier;

/// A labeled DensePolynomial with coefficients over `F`
pub type LabeledPolynomial<F> = ark_poly_commit::LabeledPolynomial<F, DensePolynomial<F>>;

/// The algebraic holographic proof defined in [CHMMVW19](https://eprint.iacr.org/2019/1047).
/// Currently, this AHP only supports inputs of size one
/// less than a power of 2 (i.e., of the form 2^n - 1).
///
/// # Notation differences from the paper (marlin_PHP.pdf)
///
/// | Paper                       | Code                  | Notes                                                       |
/// |-----------------------------|-----------------------|-------------------------------------------------------------|
/// | `ẑ_C(X)`                    | `z_c`                 | LDE of `Cz`; C is the output selector matrix                |
/// | `ẑ_B(X)`                    | `z_b`                 | LDE of `Bz`                                                 |
/// | `ŵ(X)`                      | `w`                   | LDE of the witness, divided by `v_X · v_Y`                  |
/// | `ŷ(X)`                      | `y`                   | LDE of the public output vector                             |
/// | `z_H[≤t](X)`                | `v_X` (domain X)      | Vanishing poly over first `t` elements of H (public inputs) |
/// | `z_H[≥n−s+1](X)`            | `v_Y` (computed)      | Vanishing poly over last `s` elements of H (public outputs) |
/// | `u_H(α, X)` (bivariate)     | `r_alpha_at_beta`     | Unnormalized bivariate Lagrange poly, evaluated at `(α, β)` |
/// | `Φ(X, Y)`                   | —                     | `Σ_{M∈{A,B}} η_M M̂*(X,Y) + σ·z_H(Y)`                        |
/// | `γ = Φ(α, β)`               | `gamma_claim`         | Inner sum; sent by prover in round 3                        |
/// | `ρ_t(X)`                    | `rho_t`               | Degree-1 blinding poly for `t(X)`                           |
/// | `t(X)`                      | `t`                   | `Σ η_M M̂*(α,X) + ρ_t(X)·z_H(X)`                             |
/// | `η` (zero-check randomizer) | `eta`                 | First field in `VerifierFirstMsg`; not in paper's verifier msg but needed for `t − Φ` check |
/// | `β'` (inner sumcheck point) | `zeta`                | Paper uses `β'`; code uses `zeta`                           |
/// | `f_{β'}` / `f(β')`          | `f_at_zeta`           | Evaluation of the inner sumcheck witness poly               |
/// | `δ` (KZG batching challenge)| not yet implemented   | Paper batches evaluation claims with `δ`                    |
/// | `val_{A*}`, `val_{B*}`      | `a_val`, `b_val`      | Arithmetization of `A*` and `B*`                            |
pub struct AHPForR1CS<F: Field> {
    field: PhantomData<F>,
}

impl<F: PrimeField> AHPForR1CS<F> {
    /// The labels for the polynomials output by the AHP indexer.
    #[rustfmt::skip]
    pub const INDEXER_POLYNOMIALS: [&'static str; 5] = [
        // Polynomials for M
        "row", "col", "a_val", "b_val", "row_col",
    ];

    /// The labels for the polynomials output by the AHP prover.
    #[rustfmt::skip]
    pub const PROVER_POLYNOMIALS: [&'static str; 12] = [
        // First round
        "w", "z_a", "z_b", "z_c", "s_1", "s_2",
        // Second round
        "t", "g_1", "h_1",
        // Third round
        "f",
        // Fourth round
        "g_2", "h_2",
    ];

    /// THe linear combinations that are statically known to evaluate to zero.
    pub const LC_WITH_ZERO_EVAL: [&'static str; 2] = ["inner_sumcheck", "outer_sumcheck"];

    pub(crate) fn polynomial_labels() -> impl Iterator<Item = String> {
        Self::INDEXER_POLYNOMIALS
            .iter()
            .chain(&Self::PROVER_POLYNOMIALS)
            .map(|s| s.to_string())
    }

    /// Check that the (formatted) public input is of the form 2^n for some integer n.
    pub fn num_formatted_public_inputs_is_admissible(num_inputs: usize) -> bool {
        num_inputs.count_ones() == 1
    }

    /// Check that the (formatted) public input is of the form 2^n for some integer n.
    pub fn formatted_public_input_is_admissible(input: &[F]) -> bool {
        Self::num_formatted_public_inputs_is_admissible(input.len())
    }

    /// The maximum degree of polynomials produced by the indexer and prover
    /// of this protocol.
    /// The number of the variables must include the "one" variable. That is, it
    /// must be with respect to the number of formatted public inputs.
    pub fn max_degree(
        num_constraints: usize,
        num_variables: usize,
        num_non_zero: usize,
    ) -> Result<usize, Error> {
        let padded_matrix_dim =
            constraint_systems::padded_matrix_dim(num_variables, num_constraints);
        let zk_bound = 1;
        let domain_h_size = GeneralEvaluationDomain::<F>::compute_size_of_domain(padded_matrix_dim)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_k_size = GeneralEvaluationDomain::<F>::compute_size_of_domain(num_non_zero)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        Ok(*[
            2 * domain_h_size + zk_bound - 2,
            domain_h_size + 1, // s_1
            domain_k_size + 1, // s_2
            domain_h_size,
            domain_h_size,
            domain_k_size, // f (blinded with v_K, degree |K|)
            domain_k_size - 1,
        ]
        .iter()
        .max()
        .unwrap())
    }

    /// Get all the strict degree bounds enforced in the AHP.
    pub fn get_degree_bounds(info: &indexer::IndexInfo<F>) -> [usize; 2] {
        let mut degree_bounds = [0usize; 2];
        let num_constraints = info.num_constraints;
        let num_non_zero = info.num_non_zero;
        let h_size = GeneralEvaluationDomain::<F>::compute_size_of_domain(num_constraints).unwrap();
        let k_size = GeneralEvaluationDomain::<F>::compute_size_of_domain(num_non_zero).unwrap();

        degree_bounds[0] = h_size - 2;
        degree_bounds[1] = k_size - 1; // g_2 degree bound (f is blinded with v_K)
        degree_bounds
    }

    /// Construct the linear combinations that are checked by the AHP.
    #[allow(non_snake_case)]
    pub fn construct_linear_combinations<E>(
        public_input: &[F],
        public_output: &[F],
        evals: &E,
        state: &verifier::VerifierState<F>,
    ) -> Result<Vec<LinearCombination<F>>, Error>
    where
        E: EvaluationsProvider<F>,
    {
        let domain_h = state.domain_h;
        let domain_k = state.domain_k;
        let k_size = domain_k.size_as_field_element();

        let public_input = constraint_systems::format_public_input(public_input);
        if !Self::formatted_public_input_is_admissible(&public_input) {
            return Err(Error::InvalidPublicInputLength);
        }
        let x_domain = GeneralEvaluationDomain::new(public_input.len())
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let first_round_msg = state.first_round_msg.unwrap();
        let eta = first_round_msg.eta; // randomizer for the t − Φ(α,·) zero-check
        let alpha = first_round_msg.alpha;
        let eta_a = first_round_msg.eta_a;
        let eta_b = first_round_msg.eta_b;
        let eta_c = first_round_msg.eta_c; // randomizer for the z_A·z_B − z_C check

        let beta = state.second_round_msg.unwrap().beta;
        let zeta = state.third_round_msg.unwrap().zeta;
        // γ = Φ(α,β): the inner sum claimed by the prover in round 3.
        let gamma_claim = state.gamma_claim.unwrap();

        let mut linear_combinations = Vec::new();

        // ----------------------------------------------------------------
        // Outer sumcheck (evaluated at β):
        //
        //   s_1(β) + r(α,β)·(η_A·z_A(β) + η_B·z_B(β)) − t(β)·ẑ(β)
        //   − v_H(β)·h_1(β) − β·g_1(β)
        //   + η_C·(z_A(β)·z_B(β) − z_C(β))
        //   + η·(t(β) − γ) = 0
        //
        // where ẑ(β) = w(β)·v_X(β)·v_Y(β) + x̂(β) + ŷ(β).
        // ----------------------------------------------------------------

        let z_b = LinearCombination::new("z_b", vec![(F::one(), "z_b")]);
        let g_1 = LinearCombination::new("g_1", vec![(F::one(), "g_1")]);
        let t = LinearCombination::new("t", vec![(F::one(), "t")]);

        // r(α,β) = unnormalized bivariate Lagrange polynomial; used in the lincheck.
        let r_alpha_at_beta = domain_h.eval_unnormalized_bivariate_lagrange_poly(alpha, beta);
        let v_H_at_alpha = domain_h.evaluate_vanishing_polynomial(alpha);
        let v_H_at_beta = domain_h.evaluate_vanishing_polynomial(beta);
        // v_X(β): vanishing polynomial over the public input domain X.
        let v_X_at_beta = x_domain.evaluate_vanishing_polynomial(beta);

        // v_Y(β): vanishing polynomial over the s output H positions, evaluated at β.
        // Outputs are the last s actual witness variables, which occupy the last s
        // interleaved witness slots used by the witness assignment. Any remaining
        // witness slots in H come only from domain padding.
        let s = state.num_output_variables;
        let h_elems: Vec<F> = domain_h.elements().collect();
        let n = domain_h.size();
        let x_domain = GeneralEvaluationDomain::new(state.num_instance_variables)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let ratio = n / x_domain.size();
        let witness_h_positions: Vec<usize> = (0..n).filter(|k| k % ratio != 0).collect();
        let num_witness_variables = state.num_variables - state.num_instance_variables;
        let used_witness_h_positions = &witness_h_positions[..num_witness_variables];
        let output_h_positions: Vec<usize> = if s > 0 {
            used_witness_h_positions[used_witness_h_positions.len() - s..].to_vec()
        } else {
            vec![]
        };
        let output_h_roots: Vec<F> = output_h_positions.iter().map(|&k| h_elems[k]).collect();
        let v_Y_at_beta: F = output_h_roots.iter().map(|&h| beta - h).product();

        let z_c = LinearCombination::new("z_c", vec![(F::one(), "z_c")]);
        let z_b_at_beta = evals.get_lc_eval(&z_b, beta)?;
        // ŷ(β): computed directly from public output values via Lagrange interpolation
        // over the s output positions of H — no commitment needed.
        let public_input_len = public_input.len();
        let public_output_len = public_output.len();
        let y_at_beta: F = output_h_roots
            .iter()
            .zip(public_output)
            .map(|(&h_i, &y_i)| {
                let lagrange_i: F = output_h_roots
                    .iter()
                    .filter(|&&h_j| h_j != h_i)
                    .map(|&h_j| (beta - h_j) / (h_i - h_j))
                    .product();
                y_i * lagrange_i
            })
            .sum();
        let y_at_input_roots: Vec<F> = (0..state.num_instance_variables)
            .map(|i| {
                let x_root = h_elems[domain_h.reindex_by_subdomain(x_domain, i)];
                output_h_roots
                    .iter()
                    .zip(public_output.iter())
                    .map(|(&h_i, &y_i)| {
                        let lagrange_i: F = output_h_roots
                            .iter()
                            .filter(|&&h_j| h_j != h_i)
                            .map(|&h_j| (x_root - h_j) / (h_i - h_j))
                            .product();
                        y_i * lagrange_i
                    })
                    .sum()
            })
            .collect();
        let t_at_beta = evals.get_lc_eval(&t, beta)?;
        let g_1_at_beta = evals.get_lc_eval(&g_1, beta)?;

        // x̂(β): evaluation of the public input polynomial. With public outputs enabled,
        // x is embedded over the input slots of H so that it vanishes on output slots.
        let x_at_beta = if s > 0 {
            let lagrange_on_h = domain_h.evaluate_all_lagrange_coefficients(beta);
            (0..state.num_instance_variables)
                .map(|i| {
                    let idx = domain_h.reindex_by_subdomain(x_domain, i);
                    lagrange_on_h[idx] * (public_input[i] - y_at_input_roots[i])
                })
                .sum()
        } else {
            x_domain
                .evaluate_all_lagrange_coefficients(beta)
                .into_iter()
                .zip(public_input.iter())
                .map(|(l, x)| l * x)
                .fold(F::zero(), |x, y| x + &y)
        };

        #[rustfmt::skip]
        let outer_sumcheck = LinearCombination::new(
            "outer_sumcheck",
            vec![
                // Blinding polynomial s_1
                (F::one(), "s_1".into()),

                // r(α,β)·(η_A·z_A(β) + η_B·z_B(β)): lincheck for A and B
                (r_alpha_at_beta * eta_a, "z_a".into()),
                (r_alpha_at_beta * eta_b * z_b_at_beta, LCTerm::One),

                // −t(β)·ẑ(β) where ẑ(β) = w(β)·v_X(β)·v_Y(β) + x̂(β) + ŷ(β)
                (-t_at_beta * v_X_at_beta * v_Y_at_beta, "w".into()),
                (-t_at_beta * x_at_beta, LCTerm::One),
                (-t_at_beta * y_at_beta, LCTerm::One),

                // Outer sumcheck quotient: −v_H(β)·h_1(β) − β·g_1(β)
                (-v_H_at_beta, "h_1".into()),
                (-beta * g_1_at_beta, LCTerm::One),

                // η_C·(z_A(β)·z_B(β) − z_C(β)): enforces A·z ∘ B·z = C·z
                (eta_c * z_b_at_beta, "z_a".into()),
                (-eta_c, "z_c".into()),

                // η·(t(β) − γ): t − Φ(α,·) vanishes on H, folded in via randomizer η
                (eta, "t".into()),
                (-eta * gamma_claim, LCTerm::One),
            ],
        );
        debug_assert!(evals.get_lc_eval(&outer_sumcheck, beta)?.is_zero());

        linear_combinations.push(z_b);
        linear_combinations.push(z_c);
        linear_combinations.push(g_1);
        linear_combinations.push(t);
        linear_combinations.push(outer_sumcheck);

        // ----------------------------------------------------------------
        // Inner sumcheck (evaluated at ζ):
        //
        //   a(ζ) − f(ζ)·b(ζ) − v_K(ζ)·h_2(ζ)
        //   + ζ·(s_2(ζ)·v_H(β) + f(ζ) − ζ·g_2(ζ) − γ/|K|) = 0
        //
        // where a(X) = v_H(α)·v_H(β)·(η_A·a_val(X) + η_B·b_val(X))
        //       b(X) = αβ − α·row(X) − β·col(X) + row_col(X)
        // C excluded from a(X): val_C not committed (C is a public selector).
        // ----------------------------------------------------------------

        let beta_alpha = beta * alpha;
        let f_lc = LinearCombination::new("f", vec![(F::one(), "f")]);
        let g_2 = LinearCombination::new("g_2", vec![(F::one(), "g_2")]);
        let s_2 = LinearCombination::new("s_2", vec![(F::one(), "s_2")]);

        let f_at_zeta = evals.get_lc_eval(&f_lc, zeta)?;
        let g_2_at_zeta = evals.get_lc_eval(&g_2, zeta)?;
        let s_2_at_zeta = evals.get_lc_eval(&s_2, zeta)?;

        let v_K_at_zeta = domain_k.evaluate_vanishing_polynomial(zeta);

        // a(ζ) = v_H(α)·v_H(β)·(η_A·a_val(ζ) + η_B·b_val(ζ))
        let mut a = LinearCombination::new("a_poly", vec![(eta_a, "a_val"), (eta_b, "b_val")]);
        a *= v_H_at_alpha * v_H_at_beta;

        // b(ζ) = αβ − α·row(ζ) − β·col(ζ) + row_col(ζ)  (the denominator of the sumcheck fraction)
        // Multiplied by f(ζ) so the inner sumcheck is a single LC evaluated at ζ.
        let mut b = LinearCombination::new(
            "denom",
            vec![
                (beta_alpha, LCTerm::One),
                (-alpha, "row".into()),
                (-beta, "col".into()),
                (F::one(), "row_col".into()),
            ],
        );
        b *= f_at_zeta;

        // ζ-correction enforces the low-degree condition on f and the sumcheck equation:
        //   s_2(ζ)·v_H(β) + f(ζ) − ζ·g_2(ζ) − γ/|K| = 0
        // Multiplied by ζ so it can be added to the inner_sumcheck LC.
        let zeta_correction = zeta
            * (s_2_at_zeta * v_H_at_beta + f_at_zeta - zeta * g_2_at_zeta - gamma_claim / k_size);

        let mut inner_sumcheck = a;
        inner_sumcheck -= &b;
        inner_sumcheck -= &LinearCombination::new("h_2", vec![(v_K_at_zeta, "h_2")]);
        inner_sumcheck +=
            &LinearCombination::new("zeta_correction", vec![(zeta_correction, LCTerm::One)]);

        inner_sumcheck.label = "inner_sumcheck".into();
        debug_assert!(evals.get_lc_eval(&inner_sumcheck, zeta)?.is_zero());

        linear_combinations.push(f_lc);
        linear_combinations.push(g_2);
        linear_combinations.push(s_2);
        linear_combinations.push(inner_sumcheck);

        linear_combinations.sort_by(|a, b| a.label.cmp(&b.label));
        Ok(linear_combinations)
    }
}

/// Abstraction that provides evaluations of (linear combinations of) polynomials
///
/// Intended to provide a common interface for both the prover and the verifier
/// when constructing linear combinations via `AHPForR1CS::construct_linear_combinations`.
pub trait EvaluationsProvider<F: Field> {
    /// Get the evaluation of linear combination `lc` at `point`.
    fn get_lc_eval(&self, lc: &LinearCombination<F>, point: F) -> Result<F, Error>;
}

impl<'a, F: Field> EvaluationsProvider<F> for ark_poly_commit::Evaluations<F, F> {
    fn get_lc_eval(&self, lc: &LinearCombination<F>, point: F) -> Result<F, Error> {
        let key = (lc.label.clone(), point);
        self.get(&key)
            .map(|v| *v)
            .ok_or(Error::MissingEval(lc.label.clone()))
    }
}

impl<F: Field, T: Borrow<LabeledPolynomial<F>>> EvaluationsProvider<F> for Vec<T> {
    fn get_lc_eval(&self, lc: &LinearCombination<F>, point: F) -> Result<F, Error> {
        let mut eval = F::zero();
        for (coeff, term) in lc.iter() {
            let value = if let LCTerm::PolyLabel(label) = term {
                self.iter()
                    .find(|p| {
                        let p: &LabeledPolynomial<F> = (*p).borrow();
                        p.label() == label
                    })
                    .ok_or(Error::MissingEval(format!(
                        "Missing {} for {}",
                        label, lc.label
                    )))?
                    .borrow()
                    .evaluate(&point)
            } else {
                assert!(term.is_one());
                F::one()
            };
            eval += *coeff * value
        }
        Ok(eval)
    }
}

/// Describes the failure modes of the AHP scheme.
#[derive(Debug)]
pub enum Error {
    /// During verification, a required evaluation is missing
    MissingEval(String),
    /// The number of public inputs is incorrect.
    InvalidPublicInputLength,
    /// The instance generated during proving does not match that in the index.
    InstanceDoesNotMatchIndex,
    /// Currently we only support square constraint matrices.
    NonSquareMatrix,
    /// An error occurred during constraint generation.
    ConstraintSystemError(SynthesisError),
}

impl From<SynthesisError> for Error {
    fn from(other: SynthesisError) -> Self {
        Error::ConstraintSystemError(other)
    }
}

/// The derivative of the vanishing polynomial
pub trait UnnormalizedBivariateLagrangePoly<F: ark_ff::FftField> {
    /// Evaluate the polynomial
    fn eval_unnormalized_bivariate_lagrange_poly(&self, x: F, y: F) -> F;

    /// Evaluate over a batch of inputs
    fn batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(&self, x: F) -> Vec<F>;

    /// Evaluate the magic polynomial over `self`
    fn batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs(&self) -> Vec<F>;
}

impl<F: PrimeField> UnnormalizedBivariateLagrangePoly<F> for GeneralEvaluationDomain<F> {
    fn eval_unnormalized_bivariate_lagrange_poly(&self, x: F, y: F) -> F {
        if x != y {
            (self.evaluate_vanishing_polynomial(x) - self.evaluate_vanishing_polynomial(y))
                / (x - y)
        } else {
            self.size_as_field_element() * x.pow(&[(self.size() - 1) as u64])
        }
    }

    fn batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(&self, x: F) -> Vec<F> {
        let vanish_x = self.evaluate_vanishing_polynomial(x);
        let mut inverses: Vec<F> = self.elements().map(|y| x - y).collect();
        ark_ff::batch_inversion(&mut inverses);

        cfg_iter_mut!(inverses).for_each(|denominator| *denominator *= vanish_x);
        inverses
    }

    fn batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs(&self) -> Vec<F> {
        let mut elems: Vec<F> = self
            .elements()
            .map(|e| e * self.size_as_field_element())
            .collect();
        elems[1..].reverse();
        elems
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{One, UniformRand, Zero};
    use ark_poly::{
        univariate::{DenseOrSparsePolynomial, DensePolynomial},
        Polynomial, UVPolynomial,
    };

    #[test]
    fn domain_unnormalized_bivariate_lagrange_poly() {
        for domain_size in 1..10 {
            let domain = GeneralEvaluationDomain::<Fr>::new(1 << domain_size).unwrap();
            let manual: Vec<_> = domain
                .elements()
                .map(|elem| domain.eval_unnormalized_bivariate_lagrange_poly(elem, elem))
                .collect();
            let fast = domain.batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs();
            assert_eq!(fast, manual);
        }
    }

    #[test]
    fn domain_unnormalized_bivariate_lagrange_poly_diff_inputs() {
        let rng = &mut ark_std::test_rng();
        for domain_size in 1..10 {
            let domain = GeneralEvaluationDomain::<Fr>::new(1 << domain_size).unwrap();
            let x = Fr::rand(rng);
            let manual: Vec<_> = domain
                .elements()
                .map(|y| domain.eval_unnormalized_bivariate_lagrange_poly(x, y))
                .collect();
            let fast = domain.batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(x);
            assert_eq!(fast, manual);
        }
    }

    #[test]
    fn test_summation() {
        let rng = &mut ark_std::test_rng();
        let size = 1 << 4;
        let domain = GeneralEvaluationDomain::<Fr>::new(1 << 4).unwrap();
        let size_as_fe = domain.size_as_field_element();
        let poly = DensePolynomial::rand(size, rng);

        let mut sum: Fr = Fr::zero();
        for eval in domain.elements().map(|e| poly.evaluate(&e)) {
            sum += eval;
        }
        let first = poly.coeffs[0] * size_as_fe;
        let last = *poly.coeffs.last().unwrap() * size_as_fe;
        println!("sum: {:?}", sum);
        println!("a_0: {:?}", first);
        println!("a_n: {:?}", last);
        println!("first + last: {:?}\n", first + last);
        assert_eq!(sum, first + last);
    }

    #[test]
    fn test_alternator_polynomial() {
        use ark_poly::Evaluations;
        let domain_k = GeneralEvaluationDomain::<Fr>::new(1 << 4).unwrap();
        let domain_h = GeneralEvaluationDomain::<Fr>::new(1 << 3).unwrap();
        let domain_h_elems = domain_h
            .elements()
            .collect::<std::collections::HashSet<_>>();
        let alternator_poly_evals = domain_k
            .elements()
            .map(|e| {
                if domain_h_elems.contains(&e) {
                    Fr::one()
                } else {
                    Fr::zero()
                }
            })
            .collect();
        let v_k: DenseOrSparsePolynomial<_> = domain_k.vanishing_polynomial().into();
        let v_h: DenseOrSparsePolynomial<_> = domain_h.vanishing_polynomial().into();
        let (divisor, remainder) = v_k.divide_with_q_and_r(&v_h).unwrap();
        assert!(remainder.is_zero());
        println!("Divisor: {:?}", divisor);
        println!(
            "{:#?}",
            divisor
                .coeffs
                .iter()
                .filter_map(|f| if !f.is_zero() {
                    Some(f.into_repr())
                } else {
                    None
                })
                .collect::<Vec<_>>()
        );

        for e in domain_h.elements() {
            println!("{:?}", divisor.evaluate(&e));
        }
        // Let p = v_K / v_H;
        // The alternator polynomial is p * t, where t is defined as
        // the LDE of p(h)^{-1} for all h in H.
        //
        // Because for each h in H, p(h) equals a constant c, we have that t
        // is the constant polynomial c^{-1}.
        //
        // Q: what is the constant c? Why is p(h) constant? What is the easiest
        // way to calculate c?
        let alternator_poly =
            Evaluations::from_vec_and_domain(alternator_poly_evals, domain_k).interpolate();
        let (quotient, remainder) = DenseOrSparsePolynomial::from(alternator_poly.clone())
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(divisor))
            .unwrap();
        assert!(remainder.is_zero());
        println!("quotient: {:?}", quotient);
        println!(
            "{:#?}",
            quotient
                .coeffs
                .iter()
                .filter_map(|f| if !f.is_zero() {
                    Some(f.into_repr())
                } else {
                    None
                })
                .collect::<Vec<_>>()
        );

        println!("{:?}", alternator_poly);
    }
}
