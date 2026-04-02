#![allow(non_snake_case)]

use crate::ahp::indexer::IndexInfo;
use crate::ahp::*;
use ark_std::rand::RngCore;

use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_poly_commit::QuerySet;

/// State of the AHP verifier
pub struct VerifierState<F: PrimeField> {
    pub(crate) domain_h: GeneralEvaluationDomain<F>,
    pub(crate) domain_k: GeneralEvaluationDomain<F>,

    pub(crate) first_round_msg: Option<VerifierFirstMsg<F>>,
    pub(crate) second_round_msg: Option<VerifierSecondMsg<F>>,
    pub(crate) third_round_msg: Option<VerifierThirdMsg<F>>,

    /// gamma_claim: the inner sum γ = Φ(α,β) sent by the prover in round 3
    pub(crate) gamma_claim: Option<F>,

    /// Number of public output variables (last s witness variables by convention).
    pub(crate) num_output_variables: usize,
    /// Total number of variables (= num_constraints after squaring).
    pub(crate) num_variables: usize,
    /// Number of formatted public input variables (size of domain X).
    pub(crate) num_instance_variables: usize,
}

/// Third verifier message.
#[derive(Copy, Clone)]
pub struct VerifierThirdMsg<F> {
    /// Query point for the inner sumcheck polynomials.
    pub zeta: F,
}

/// First message of the verifier.
#[derive(Copy, Clone)]
pub struct VerifierFirstMsg<F> {
    /// Randomizer for the t(X) - Φ(α,X) zero-check.
    pub eta: F,
    /// Query for the random polynomial.
    pub alpha: F,
    /// Randomizer for the lincheck for `A`.
    pub eta_a: F,
    /// Randomizer for the lincheck for `B`.
    pub eta_b: F,
    /// Randomizer for the lincheck for `C`.
    pub eta_c: F,
}

/// Second verifier message.
#[derive(Copy, Clone)]
pub struct VerifierSecondMsg<F> {
    /// Query for the second round of polynomials.
    pub beta: F,
}

impl<F: PrimeField> AHPForR1CS<F> {
    /// Output the first message and next round state.
    pub fn verifier_first_round<R: RngCore>(
        index_info: IndexInfo<F>,
        rng: &mut R,
    ) -> Result<(VerifierFirstMsg<F>, VerifierState<F>), Error> {
        let num_output_variables = index_info.num_output_variables;
        if index_info.num_constraints != index_info.num_variables {
            return Err(Error::NonSquareMatrix);
        }

        let domain_h = GeneralEvaluationDomain::new(index_info.num_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let domain_k = GeneralEvaluationDomain::new(index_info.num_non_zero)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let eta = F::rand(rng);
        let alpha = domain_h.sample_element_outside_domain(rng);
        let eta_a = F::rand(rng);
        let eta_b = F::rand(rng);
        let eta_c = F::rand(rng);

        let msg = VerifierFirstMsg {
            eta,
            alpha,
            eta_a,
            eta_b,
            eta_c,
        };

        let new_state = VerifierState {
            domain_h,
            domain_k,
            first_round_msg: Some(msg),
            second_round_msg: None,
            third_round_msg: None,
            gamma_claim: None,
            num_output_variables,
            num_variables: index_info.num_variables,
            num_instance_variables: index_info.num_instance_variables,
        };

        Ok((msg, new_state))
    }

    /// Output the second message and next round state.
    pub fn verifier_second_round<R: RngCore>(
        mut state: VerifierState<F>,
        rng: &mut R,
    ) -> (VerifierSecondMsg<F>, VerifierState<F>) {
        let beta = state.domain_h.sample_element_outside_domain(rng);
        let msg = VerifierSecondMsg { beta };
        state.second_round_msg = Some(msg);

        (msg, state)
    }

    /// Output the third message and next round state.
    /// Absorbs the prover's gamma_claim (γ = Φ(α,β)) and samples zeta.
    pub fn verifier_third_round<R: RngCore>(
        mut state: VerifierState<F>,
        gamma_claim: F,
        rng: &mut R,
    ) -> (VerifierThirdMsg<F>, VerifierState<F>) {
        state.gamma_claim = Some(gamma_claim);
        let zeta = state.domain_k.sample_element_outside_domain(rng);
        let msg = VerifierThirdMsg { zeta };
        state.third_round_msg = Some(msg);
        (msg, state)
    }

    /// Output the query state and next round state.
    pub fn verifier_query_set<'a, R: RngCore>(
        state: VerifierState<F>,
        _: &'a mut R,
    ) -> (QuerySet<F>, VerifierState<F>) {
        let beta = state.second_round_msg.unwrap().beta;

        let zeta = state.third_round_msg.unwrap().zeta;

        let mut query_set = QuerySet::new();

        // Outer sumcheck (evaluated at β):
        //
        //   s_1(β) + r(α,β)·(η_A·z_A(β) + η_B·z_B(β)) - t(β)·ẑ(β)
        //   - v_H(β)·h_1(β) - β·g_1(β)
        //   + η_C·(z_A(β)·z_B(β) - z_C(β))
        //   + η·(t(β) - γ)
        // = 0
        //
        // where ẑ(β) = w(β)·v_X(β)·v_Y(β) + x̂(β) + ŷ(β)
        // and γ = Φ(α,β) is the inner sum sent by the prover in round 3.
        //
        // Polynomials evaluated at β: g_1, z_b, z_c, y, t, outer_sumcheck.
        query_set.insert(("g_1".into(), ("beta".into(), beta)));
        query_set.insert(("z_b".into(), ("beta".into(), beta)));
        query_set.insert(("z_c".into(), ("beta".into(), beta)));
        query_set.insert(("t".into(), ("beta".into(), beta)));
        query_set.insert(("outer_sumcheck".into(), ("beta".into(), beta)));

        // Inner sumcheck (evaluated at ζ):
        //
        //   a(ζ) - f(ζ)·b(ζ) - v_K(ζ)·h_2(ζ)
        //   + ζ·(s_2(ζ)·v_H(β) + f(ζ) - ζ·g_2(ζ) - γ/|K|)
        // = 0
        //
        // where
        //   a(X) = v_H(α)·v_H(β)·(η_A·a_val(X) + η_B·b_val(X))
        //   b(X) = αβ - α·row(X) - β·col(X) + row_col(X)
        //   f(X) = the prover's witness polynomial for the sumcheck quotient
        //
        // C is excluded from a(X) since val_C is not committed (C is a public selector matrix).
        //
        // Polynomials evaluated at ζ: f, g_2, s_2, inner_sumcheck.
        query_set.insert(("f".into(), ("zeta".into(), zeta)));
        query_set.insert(("g_2".into(), ("zeta".into(), zeta)));
        query_set.insert(("s_2".into(), ("zeta".into(), zeta)));
        query_set.insert(("inner_sumcheck".into(), ("zeta".into(), zeta)));

        (query_set, state)
    }
}
