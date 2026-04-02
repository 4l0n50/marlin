#![allow(non_snake_case)]

use crate::ahp::indexer::*;
use crate::ahp::verifier::*;
use crate::ahp::*;

use crate::ahp::constraint_systems::{
    make_matrices_square_for_prover, pad_input_for_indexer_and_prover, unformat_public_input,
};
use crate::{ToString, Vec};
use ark_ff::{Field, PrimeField, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain, Polynomial, UVPolynomial,
};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisError,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use ark_std::rand::RngCore;
use ark_std::{
    cfg_into_iter, cfg_iter, cfg_iter_mut,
    io::{Read, Write},
};

/// State for the AHP prover.
pub struct ProverState<'a, F: PrimeField> {
    formatted_input_assignment: Vec<F>,
    witness_assignment: Vec<F>,
    /// Az
    z_a: Option<Vec<F>>,
    /// Bz
    z_b: Option<Vec<F>>,
    /// Cz
    z_c: Option<Vec<F>>,
    /// query bound b
    zk_bound: usize,

    w_poly: Option<LabeledPolynomial<F>>,
    mz_polys: Option<(LabeledPolynomial<F>, LabeledPolynomial<F>)>,
    z_c_poly: Option<LabeledPolynomial<F>>,
    /// y_poly: LDE of the public output values over their output positions in H
    /// (not committed; verifier computes y(β) directly)
    y_poly: Option<DensePolynomial<F>>,
    /// Public output values (last s witness variables); stored so the prover-side
    /// output embedding can be reconstructed consistently in later rounds.
    pub output_assignment: Vec<F>,
    /// v_Y_poly: vanishing polynomial over the output positions in H
    v_Y_poly: Option<DensePolynomial<F>>,

    index: &'a Index<F>,

    /// the random values sent by the verifier in the first round
    verifier_first_msg: Option<VerifierFirstMsg<F>>,

    /// s_1: the blinding polynomial for the first sumcheck (degree |H|+1)
    s_1: Option<LabeledPolynomial<F>>,

    /// s_2: the blinding polynomial for the second sumcheck (degree |K|+1)
    s_2: Option<LabeledPolynomial<F>>,

    /// σ: the blinding scalar used to construct s_2
    sigma: Option<F>,

    /// ρ_t: degree-1 blinding polynomial for t; stored so round 3 can recover γ = t(β) - ρ_t(β)·v_H(β)
    rho_t: Option<DensePolynomial<F>>,

    /// f: the inner sumcheck polynomial, computed in round 3, used in round 4
    f_poly: Option<LabeledPolynomial<F>>,

    /// γ = Φ(α,β) = t(β): the inner sum claim sent as the round 3 prover message
    gamma_claim: Option<F>,

    /// t polynomial (blinded), stored in round 2 so round 3 can evaluate t(β)
    t_poly: Option<DensePolynomial<F>>,

    /// β: the verifier's second challenge, stored in round 2 for use in rounds 3 and 4
    beta: Option<F>,

    /// domain X, sized for the public input
    domain_x: GeneralEvaluationDomain<F>,

    /// domain H, sized for constraints
    domain_h: GeneralEvaluationDomain<F>,

    /// domain K, sized for matrix nonzero elements
    domain_k: GeneralEvaluationDomain<F>,
}

impl<'a, F: PrimeField> ProverState<'a, F> {
    /// Get the public input.
    pub fn public_input(&self) -> Vec<F> {
        unformat_public_input(&self.formatted_input_assignment)
    }
}

/// Each prover message that is not a list of oracles is a list of field elements.
#[derive(Clone)]
pub enum ProverMsg<F: Field> {
    /// Some rounds, the prover sends only oracles. (This is actually the case for all
    /// rounds in Marlin.)
    EmptyMessage,
    /// Otherwise, it's one or more field elements.
    FieldElements(Vec<F>),
}

impl<F: Field> ark_ff::ToBytes for ProverMsg<F> {
    fn write<W: Write>(&self, w: W) -> ark_std::io::Result<()> {
        match self {
            ProverMsg::EmptyMessage => Ok(()),
            ProverMsg::FieldElements(field_elems) => field_elems.write(w),
        }
    }
}

impl<F: Field> CanonicalSerialize for ProverMsg<F> {
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        let res: Option<Vec<F>> = match self {
            ProverMsg::EmptyMessage => None,
            ProverMsg::FieldElements(v) => Some(v.clone()),
        };
        res.serialize(&mut writer)
    }

    fn serialized_size(&self) -> usize {
        let res: Option<Vec<F>> = match self {
            ProverMsg::EmptyMessage => None,
            ProverMsg::FieldElements(v) => Some(v.clone()),
        };
        res.serialized_size()
    }

    fn serialize_unchecked<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        let res: Option<Vec<F>> = match self {
            ProverMsg::EmptyMessage => None,
            ProverMsg::FieldElements(v) => Some(v.clone()),
        };
        res.serialize_unchecked(&mut writer)
    }

    fn serialize_uncompressed<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        let res: Option<Vec<F>> = match self {
            ProverMsg::EmptyMessage => None,
            ProverMsg::FieldElements(v) => Some(v.clone()),
        };
        res.serialize_uncompressed(&mut writer)
    }

    fn uncompressed_size(&self) -> usize {
        let res: Option<Vec<F>> = match self {
            ProverMsg::EmptyMessage => None,
            ProverMsg::FieldElements(v) => Some(v.clone()),
        };
        res.uncompressed_size()
    }
}

impl<F: Field> CanonicalDeserialize for ProverMsg<F> {
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        let res = Option::<Vec<F>>::deserialize(&mut reader)?;

        if let Some(res) = res {
            Ok(ProverMsg::FieldElements(res))
        } else {
            Ok(ProverMsg::EmptyMessage)
        }
    }

    fn deserialize_unchecked<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        let res = Option::<Vec<F>>::deserialize_unchecked(&mut reader)?;

        if let Some(res) = res {
            Ok(ProverMsg::FieldElements(res))
        } else {
            Ok(ProverMsg::EmptyMessage)
        }
    }

    fn deserialize_uncompressed<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        let res = Option::<Vec<F>>::deserialize_uncompressed(&mut reader)?;

        if let Some(res) = res {
            Ok(ProverMsg::FieldElements(res))
        } else {
            Ok(ProverMsg::EmptyMessage)
        }
    }
}

/// The first set of prover oracles.
pub struct ProverFirstOracles<F: Field> {
    /// The LDE of `w` (internal witness, zero at x and y positions).
    pub w: LabeledPolynomial<F>,
    /// The LDE of `Az`.
    pub z_a: LabeledPolynomial<F>,
    /// The LDE of `Bz`.
    pub z_b: LabeledPolynomial<F>,
    /// The LDE of `z_C = (0, w)` (zeros at x positions, witness elsewhere).
    pub z_c: LabeledPolynomial<F>,
    /// s_1: blinding polynomial for the first sumcheck.
    pub s_1: LabeledPolynomial<F>,
    /// s_2: blinding polynomial for the second sumcheck.
    pub s_2: LabeledPolynomial<F>,
}

impl<F: Field> ProverFirstOracles<F> {
    /// Iterate over the polynomials output by the prover in the first round.
    pub fn iter(&self) -> impl Iterator<Item = &LabeledPolynomial<F>> {
        vec![
            &self.w, &self.z_a, &self.z_b, &self.z_c, &self.s_1, &self.s_2,
        ]
        .into_iter()
    }
}

/// The second set of prover oracles.
pub struct ProverSecondOracles<F: Field> {
    /// The polynomial `t` that is produced in the first round.
    pub t: LabeledPolynomial<F>,
    /// The polynomial `g` resulting from the first sumcheck.
    pub g_1: LabeledPolynomial<F>,
    /// The polynomial `h` resulting from the first sumcheck.
    pub h_1: LabeledPolynomial<F>,
}

impl<F: Field> ProverSecondOracles<F> {
    /// Iterate over the polynomials output by the prover in the second round.
    pub fn iter(&self) -> impl Iterator<Item = &LabeledPolynomial<F>> {
        vec![&self.t, &self.g_1, &self.h_1].into_iter()
    }
}

/// The third set of prover oracles.
pub struct ProverThirdOracles<F: Field> {
    /// The inner sumcheck polynomial f.
    pub f: LabeledPolynomial<F>,
}

impl<F: Field> ProverThirdOracles<F> {
    /// Iterate over the polynomials output by the prover in the third round.
    pub fn iter(&self) -> impl Iterator<Item = &LabeledPolynomial<F>> {
        vec![&self.f].into_iter()
    }
}

/// The fourth set of prover oracles.
pub struct ProverFourthOracles<F: Field> {
    /// The polynomial `g` resulting from the second sumcheck.
    pub g_2: LabeledPolynomial<F>,
    /// The polynomial `h` resulting from the second sumcheck.
    pub h_2: LabeledPolynomial<F>,
}

impl<F: Field> ProverFourthOracles<F> {
    /// Iterate over the polynomials output by the prover in the fourth round.
    pub fn iter(&self) -> impl Iterator<Item = &LabeledPolynomial<F>> {
        vec![&self.g_2, &self.h_2].into_iter()
    }
}

impl<F: PrimeField> AHPForR1CS<F> {
    /// Initialize the AHP prover.
    pub fn prover_init<'a, C: ConstraintSynthesizer<F>>(
        index: &'a Index<F>,
        c: C,
    ) -> Result<ProverState<'a, F>, Error> {
        let init_time = start_timer!(|| "AHP::Prover::Init");

        let constraint_time = start_timer!(|| "Generating constraints and witnesses");
        let pcs = ConstraintSystem::new_ref();
        pcs.set_optimization_goal(OptimizationGoal::Weight);
        pcs.set_mode(ark_relations::r1cs::SynthesisMode::Prove {
            construct_matrices: true,
        });
        c.generate_constraints(pcs.clone())?;
        end_timer!(constraint_time);

        let padding_time = start_timer!(|| "Padding matrices to make them square");
        pad_input_for_indexer_and_prover(pcs.clone());
        pcs.finalize();
        make_matrices_square_for_prover(pcs.clone());
        end_timer!(padding_time);

        let num_non_zero = index.index_info.num_non_zero;

        let (formatted_input_assignment, witness_assignment, num_constraints) = {
            let pcs = pcs.into_inner().unwrap();
            (
                pcs.instance_assignment,
                pcs.witness_assignment,
                pcs.num_constraints,
            )
        };

        let num_input_variables = formatted_input_assignment.len();
        let num_witness_variables = witness_assignment.len();
        if index.index_info.num_constraints != num_constraints
            || num_input_variables + num_witness_variables != index.index_info.num_variables
        {
            return Err(Error::InstanceDoesNotMatchIndex);
        }

        if !Self::formatted_public_input_is_admissible(&formatted_input_assignment) {
            return Err(Error::InvalidPublicInputLength);
        }

        // Perform matrix multiplications
        let inner_prod_fn = |row: &[(F, usize)]| {
            let mut acc = F::zero();
            for &(ref coeff, i) in row {
                let tmp = if i < num_input_variables {
                    formatted_input_assignment[i]
                } else {
                    witness_assignment[i - num_input_variables]
                };

                acc += &(if coeff.is_one() { tmp } else { tmp * coeff });
            }
            acc
        };

        let eval_z_a_time = start_timer!(|| "Evaluating z_A");
        let z_a = index.a.iter().map(|row| inner_prod_fn(row)).collect();
        end_timer!(eval_z_a_time);

        let eval_z_b_time = start_timer!(|| "Evaluating z_B");
        let z_b = index.b.iter().map(|row| inner_prod_fn(row)).collect();
        end_timer!(eval_z_b_time);

        let eval_z_c_time = start_timer!(|| "Evaluating z_C");
        let z_c = index.c.iter().map(|row| inner_prod_fn(row)).collect();
        end_timer!(eval_z_c_time);

        let zk_bound = 1; // One query is sufficient for our desired soundness

        let domain_h = GeneralEvaluationDomain::new(num_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let domain_k = GeneralEvaluationDomain::new(num_non_zero)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let domain_x = GeneralEvaluationDomain::new(num_input_variables)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        end_timer!(init_time);

        Ok(ProverState {
            formatted_input_assignment,
            witness_assignment,
            z_a: Some(z_a),
            z_b: Some(z_b),
            z_c: Some(z_c),
            w_poly: None,
            mz_polys: None,
            z_c_poly: None,
            y_poly: None,
            v_Y_poly: None,
            output_assignment: Vec::new(),
            zk_bound,
            index,
            verifier_first_msg: None,
            s_1: None,
            s_2: None,
            sigma: None,
            rho_t: None,
            f_poly: None,
            gamma_claim: None,
            t_poly: None,
            beta: None,
            domain_h,
            domain_k,
            domain_x,
        })
    }

    /// Output the first round message and the next state.
    pub fn prover_first_round<'a, R: RngCore>(
        mut state: ProverState<'a, F>,
        rng: &mut R,
    ) -> Result<(ProverMsg<F>, ProverFirstOracles<F>, ProverState<'a, F>), Error> {
        let round_time = start_timer!(|| "AHP::Prover::FirstRound");
        let domain_h = state.domain_h;
        let zk_bound = state.zk_bound;

        let v_H = domain_h.vanishing_polynomial().into();

        let x_time = start_timer!(|| "Computing x polynomial and evals");
        let domain_x = state.domain_x;
        let num_output = state.index.index_info.num_output_variables;
        let s = num_output;
        let n = domain_h.size();
        let ratio = n / domain_x.size();
        let h_elems: Vec<F> = domain_h.elements().collect();
        let (_x_poly, mut x_evals) = if s > 0 {
            let mut x_evals = vec![F::zero(); n];
            for (i, x_i) in state.formatted_input_assignment.iter().enumerate() {
                let idx = domain_h.reindex_by_subdomain(domain_x, i);
                x_evals[idx] = *x_i;
            }
            let x_poly =
                EvaluationsOnDomain::from_vec_and_domain(x_evals.clone(), domain_h).interpolate();
            (x_poly, x_evals)
        } else {
            let x_poly = EvaluationsOnDomain::from_vec_and_domain(
                state.formatted_input_assignment.clone(),
                domain_x,
            )
            .interpolate();
            let x_evals = domain_h.fft(&x_poly);
            (x_poly, x_evals)
        };
        end_timer!(x_time);

        // Output positions in H: the last s actual witness variables occupy the last s
        // interleaved witness slots used by the witness assignment. Any remaining
        // witness slots in H come from domain padding and are not real variables.
        let witness_h_positions: Vec<usize> = (0..n).filter(|k| k % ratio != 0).collect();
        let used_witness_h_positions = &witness_h_positions[..state.witness_assignment.len()];
        let output_h_positions: Vec<usize> = if s > 0 {
            used_witness_h_positions[used_witness_h_positions.len() - s..].to_vec()
        } else {
            vec![]
        };
        let output_h_roots: Vec<F> = output_h_positions.iter().map(|&k| h_elems[k]).collect();

        let v_Y_poly: DensePolynomial<F> = {
            let mut coeffs = vec![F::one()];
            for &root in &output_h_roots {
                let mut new_coeffs = vec![F::zero(); coeffs.len() + 1];
                for (i, c) in coeffs.iter().enumerate() {
                    new_coeffs[i + 1] += c;
                    new_coeffs[i] -= *c * root;
                }
                coeffs = new_coeffs;
            }
            DensePolynomial::from_coefficients_vec(coeffs)
        };

        // y values: last s entries of witness_assignment
        let witness = &state.witness_assignment;
        let num_witness = witness.len();
        let y_vals: Vec<F> = if s > 0 && num_witness >= s {
            witness[num_witness - s..].to_vec()
        } else {
            vec![F::zero(); s]
        };

        // y_poly: the low-degree polynomial interpolating the public outputs over
        // their output positions in H.
        let y_poly_time = start_timer!(|| "Computing y polynomial");
        let y_poly = {
            let mut acc = DensePolynomial::zero();
            for (i, (&h_i, &y_i)) in output_h_roots.iter().zip(&y_vals).enumerate() {
                let mut numer = DensePolynomial::from_coefficients_slice(&[F::one()]);
                let mut denom = F::one();
                for (j, &h_j) in output_h_roots.iter().enumerate() {
                    if i != j {
                        numer = &numer
                            * &DensePolynomial::from_coefficients_slice(&[-h_j, F::one()]);
                        denom *= h_i - h_j;
                    }
                }
                acc += &(&numer * (y_i / denom));
            }
            acc
        };
        let y_evals_on_h = y_poly.evaluate_over_domain_by_ref(domain_h).evals;
        end_timer!(y_poly_time);

        if s > 0 {
            for i in 0..state.formatted_input_assignment.len() {
                let idx = domain_h.reindex_by_subdomain(domain_x, i);
                x_evals[idx] -= y_evals_on_h[idx];
            }
        }

        // ẑ(X) = ŵ(X)·v_X(X)·v_Y(X) + x̂(X) + ŷ(X)
        // => ŵ(X) = (ẑ(X) - x̂(X) - ŷ(X)) / (v_X(X)·v_Y(X))
        let mut z_evals_on_h = vec![F::zero(); n];
        for (i, x_i) in state.formatted_input_assignment.iter().enumerate() {
            let idx = domain_h.reindex_by_subdomain(domain_x, i);
            z_evals_on_h[idx] = *x_i;
        }
        for (&idx, &w_i) in used_witness_h_positions.iter().zip(witness.iter()) {
            z_evals_on_h[idx] = w_i;
        }

        let w_poly_time = start_timer!(|| "Computing w polynomial");
        let w_poly_evals: Vec<F> = cfg_into_iter!(0..n)
            .map(|k| z_evals_on_h[k] - x_evals[k] - y_evals_on_h[k])
            .collect();

        let w_poly_blinded = &EvaluationsOnDomain::from_vec_and_domain(w_poly_evals, domain_h)
            .interpolate()
            + &(&DensePolynomial::from_coefficients_slice(&[F::rand(rng)]) * &v_H);
        use ark_poly::univariate::DenseOrSparsePolynomial;
        // Divide by v_X first (using the fast domain method), then by v_Y
        let (w_poly_div_vx, remainder_vx) =
            w_poly_blinded.divide_by_vanishing_poly(domain_x).unwrap();
        assert!(remainder_vx.is_zero());
        let (w_poly, remainder) = if s > 0 {
            DenseOrSparsePolynomial::from(w_poly_div_vx)
                .divide_with_q_and_r(&DenseOrSparsePolynomial::from(v_Y_poly.clone()))
                .unwrap()
        } else {
            (
                w_poly_div_vx,
                DensePolynomial::from_coefficients_vec(vec![]),
            )
        };
        assert!(remainder.is_zero());
        end_timer!(w_poly_time);

        let z_a_poly_time = start_timer!(|| "Computing z_A polynomial");
        let z_a = state.z_a.clone().unwrap();
        let z_a_poly = &EvaluationsOnDomain::from_vec_and_domain(z_a, domain_h).interpolate()
            + &(&DensePolynomial::from_coefficients_slice(&[F::rand(rng)]) * &v_H);
        end_timer!(z_a_poly_time);

        let z_b_poly_time = start_timer!(|| "Computing z_B polynomial");
        let z_b = state.z_b.clone().unwrap();
        let z_b_poly = &EvaluationsOnDomain::from_vec_and_domain(z_b, domain_h).interpolate()
            + &(&DensePolynomial::from_coefficients_slice(&[F::rand(rng)]) * &v_H);
        end_timer!(z_b_poly_time);

        // z_C: committed as Cz (matrix multiplication), where C is assumed to be the selector
        // c = (0,...,0,1,...,1) picking out non-input variables, so z_C = (0, w) on H.
        let z_c_poly_time = start_timer!(|| "Computing z_C polynomial");
        let z_c = state.z_c.clone().unwrap();
        let z_c_poly_r1 = &EvaluationsOnDomain::from_vec_and_domain(z_c, domain_h).interpolate()
            + &(&DensePolynomial::from_coefficients_slice(&[F::rand(rng)]) * &v_H);
        end_timer!(z_c_poly_time);

        // s_1(X) = q_{s1} * z_H(X) + X * r_{s1}  (degree |H| + 1)
        // Sum over H is zero by construction (z_H vanishes on H; X*r_{s1} sums to 0).
        let s_1_time = start_timer!(|| "Computing s_1 polynomial");
        let q_s1 = F::rand(rng);
        let r_s1 = F::rand(rng);
        let v_H_poly: DensePolynomial<F> = domain_h.vanishing_polynomial().into();
        // q_{s1} * z_H(X)
        let s_1_poly = &DensePolynomial::from_coefficients_slice(&[q_s1]) * &v_H_poly
            // + X * r_{s1}
            + DensePolynomial::from_coefficients_slice(&[F::zero(), r_s1]);
        end_timer!(s_1_time);

        // s_2(X) = q_{s2} * z_K(X) + X * r_{s2} + σ/|K|  (degree |K| + 1)
        // Sum over K equals σ: z_K vanishes on K, X*r_{s2} sums to 0, constant σ/|K| sums to σ.
        let s_2_time = start_timer!(|| "Computing s_2 polynomial");
        let domain_k = state.domain_k;
        let v_K_poly: DensePolynomial<F> = domain_k.vanishing_polynomial().into();
        let q_s2 = F::rand(rng);
        let r_s2 = F::rand(rng);
        let sigma = F::rand(rng);
        let sigma_over_k = sigma / domain_k.size_as_field_element();
        let s_2_poly = &DensePolynomial::from_coefficients_slice(&[q_s2]) * &v_K_poly
            + DensePolynomial::from_coefficients_slice(&[sigma_over_k, r_s2]);
        end_timer!(s_2_time);

        let msg = ProverMsg::EmptyMessage;

        assert!(z_a_poly.degree() < domain_h.size() + zk_bound);
        assert!(z_b_poly.degree() < domain_h.size() + zk_bound);
        assert!(z_c_poly_r1.degree() < domain_h.size() + zk_bound);
        assert!(y_poly.degree() < domain_h.size() + zk_bound);
        assert!(s_1_poly.degree() <= domain_h.size() + 1);
        assert!(s_2_poly.degree() <= domain_k.size() + 1);

        let w = LabeledPolynomial::new("w".to_string(), w_poly, None, Some(1));
        let z_a = LabeledPolynomial::new("z_a".to_string(), z_a_poly, None, Some(1));
        let z_b = LabeledPolynomial::new("z_b".to_string(), z_b_poly, None, Some(1));
        let z_c_committed = LabeledPolynomial::new("z_c".to_string(), z_c_poly_r1, None, Some(1));
        let s_1 = LabeledPolynomial::new("s_1".to_string(), s_1_poly, None, None);
        let s_2 = LabeledPolynomial::new("s_2".to_string(), s_2_poly, None, None);

        let oracles = ProverFirstOracles {
            w: w.clone(),
            z_a: z_a.clone(),
            z_b: z_b.clone(),
            z_c: z_c_committed.clone(),
            s_1: s_1.clone(),
            s_2: s_2.clone(),
        };

        state.w_poly = Some(w);
        state.mz_polys = Some((z_a, z_b));
        state.z_c_poly = Some(z_c_committed);
        state.output_assignment = y_vals;
        state.y_poly = Some(y_poly);
        state.v_Y_poly = Some(v_Y_poly);
        state.s_1 = Some(s_1);
        state.s_2 = Some(s_2);
        state.sigma = Some(sigma);
        end_timer!(round_time);

        Ok((msg, oracles, state))
    }

    fn calculate_t<'a>(
        matrices: impl Iterator<Item = &'a Matrix<F>>,
        matrix_randomizers: &[F],
        input_domain: GeneralEvaluationDomain<F>,
        domain_h: GeneralEvaluationDomain<F>,
        r_alpha_x_on_h: Vec<F>,
    ) -> DensePolynomial<F> {
        let mut t_evals_on_h = vec![F::zero(); domain_h.size()];
        for (matrix, eta) in matrices.zip(matrix_randomizers) {
            for (r, row) in matrix.iter().enumerate() {
                for (coeff, c) in row.iter() {
                    let index = domain_h.reindex_by_subdomain(input_domain, *c);
                    t_evals_on_h[index] += *eta * coeff * r_alpha_x_on_h[r];
                }
            }
        }
        EvaluationsOnDomain::from_vec_and_domain(t_evals_on_h, domain_h).interpolate()
    }

    /// Output the number of oracles sent by the prover in the first round.
    pub fn prover_num_first_round_oracles() -> usize {
        4
    }

    /// Output the degree bounds of oracles in the first round.
    pub fn prover_first_round_degree_bounds(
        _info: &IndexInfo<F>,
    ) -> impl Iterator<Item = Option<usize>> {
        vec![None; 6].into_iter()
    }

    /// Output the second round message and the next state.
    pub fn prover_second_round<'a, R: RngCore>(
        ver_message: &VerifierFirstMsg<F>,
        mut state: ProverState<'a, F>,
        zk_rng: &mut R,
    ) -> (ProverMsg<F>, ProverSecondOracles<F>, ProverState<'a, F>) {
        let round_time = start_timer!(|| "AHP::Prover::SecondRound");

        let domain_h = state.domain_h;
        let zk_bound = state.zk_bound;

        let s_1 = state
            .s_1
            .as_ref()
            .expect("ProverState should include s_1 when prover_second_round is called");
        let sigma = state
            .sigma
            .expect("ProverState should include sigma when prover_second_round is called");

        let VerifierFirstMsg {
            eta,
            alpha,
            eta_a,
            eta_b,
            eta_c,
        } = *ver_message;

        let summed_z_m_poly_time = start_timer!(|| "Compute z_m poly");
        let (z_a_poly, z_b_poly) = state.mz_polys.as_ref().unwrap();

        // summed_z_m = η_A·z_A + η_B·z_B  (C excluded; η_C enters via zero-check below)
        let summed_z_m = {
            let mut coeffs = vec![F::zero(); domain_h.size() + zk_bound];
            cfg_iter_mut!(coeffs)
                .zip(&z_a_poly.polynomial().coeffs)
                .zip(&z_b_poly.polynomial().coeffs)
                .for_each(|((c, a), b)| *c += eta_a * a + eta_b * b);
            DensePolynomial::from_coefficients_vec(coeffs)
        };
        end_timer!(summed_z_m_poly_time);

        let v_H_poly: DensePolynomial<F> = domain_h.vanishing_polynomial().into();
        // Use the committed z_c polynomial from round 1
        let z_c_poly = state
            .z_c_poly
            .as_ref()
            .expect("ProverState should include z_c_poly when prover_second_round is called")
            .polynomial()
            .clone();

        let r_alpha_x_evals_time = start_timer!(|| "Compute r_alpha_x evals");
        let r_alpha_x_evals =
            domain_h.batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(alpha);
        end_timer!(r_alpha_x_evals_time);

        let r_alpha_poly_time = start_timer!(|| "Compute r_alpha_x poly");
        let r_alpha_poly = DensePolynomial::from_coefficients_vec(domain_h.ifft(&r_alpha_x_evals));
        end_timer!(r_alpha_poly_time);

        let t_poly_time = start_timer!(|| "Compute t poly");
        // t(X) = Σ_{M∈{A,B}} η_M M̂*(α,X) + ρ_t(X)·z_H(X)
        // where ρ_t is a random degree-1 polynomial.
        // t agrees with Φ(α,X) on H but differs as a polynomial (zero-check t - Φ is folded into h_1).
        let mut t_poly = Self::calculate_t(
            vec![&state.index.a, &state.index.b].into_iter(),
            &[eta_a, eta_b],
            state.domain_x,
            state.domain_h,
            r_alpha_x_evals,
        );
        let rho_t_0 = F::rand(zk_rng);
        let rho_t_1 = F::rand(zk_rng);
        let rho_t = DensePolynomial::from_coefficients_slice(&[rho_t_0, rho_t_1]);
        t_poly += &(&rho_t * &v_H_poly);
        end_timer!(t_poly_time);
        state.rho_t = Some(rho_t.clone());

        let z_poly_time = start_timer!(|| "Compute z poly");

        let domain_x = GeneralEvaluationDomain::new(state.formatted_input_assignment.len())
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let y_poly = state.y_poly.as_ref().unwrap();
        let x_poly = if state.index.index_info.num_output_variables > 0 {
            let mut x_evals_on_h = vec![F::zero(); domain_h.size()];
            let h_elems: Vec<F> = domain_h.elements().collect();
            for (i, x_i) in state.formatted_input_assignment.iter().enumerate() {
                let idx = domain_h.reindex_by_subdomain(domain_x, i);
                x_evals_on_h[idx] = *x_i - y_poly.evaluate(&h_elems[idx]);
            }
            EvaluationsOnDomain::from_vec_and_domain(x_evals_on_h, domain_h).interpolate()
        } else {
            EvaluationsOnDomain::from_vec_and_domain(
                state.formatted_input_assignment.clone(),
                domain_x,
            )
            .interpolate()
        };
        let w_poly = state.w_poly.as_ref().unwrap();
        let v_Y_poly = state.v_Y_poly.as_ref().unwrap().clone();
        // ẑ(X) = ŵ(X)·v_X(X)·v_Y(X) + x̂(X) + ŷ(X)
        let v_X_poly: DensePolynomial<F> = domain_x.vanishing_polynomial().into();
        let z_poly = &(&(w_poly.polynomial() * &v_X_poly) * &v_Y_poly) + &(&x_poly + y_poly);
        assert!(z_poly.degree() < domain_h.size() + zk_bound);
        end_timer!(z_poly_time);

        let q_1_time = start_timer!(|| "Compute q_1 poly");

        // z_a * z_b product (degree 2(|H|+zk-1))
        let z_a_times_z_b = z_a_poly.polynomial() * z_b_poly.polynomial();

        let mul_domain_size = *[
            s_1.len(),
            r_alpha_poly.coeffs.len() + summed_z_m.coeffs.len(),
            t_poly.coeffs.len() + z_poly.len(),
            z_a_times_z_b.coeffs.len(), // for the η_C zero-check term
        ]
        .iter()
        .max()
        .unwrap();
        let mul_domain = GeneralEvaluationDomain::new(mul_domain_size)
            .expect("field is not smooth enough to construct domain");
        let mut r_alpha_evals = r_alpha_poly.evaluate_over_domain_by_ref(mul_domain);
        let summed_z_m_evals = summed_z_m.evaluate_over_domain_by_ref(mul_domain);
        let z_poly_evals = z_poly.evaluate_over_domain_by_ref(mul_domain);
        let t_poly_m_evals = t_poly.evaluate_over_domain_by_ref(mul_domain);
        let z_a_z_b_evals = z_a_times_z_b.evaluate_over_domain_by_ref(mul_domain);
        let z_c_evals_mul = z_c_poly.evaluate_over_domain_by_ref(mul_domain);

        let lin_term_poly = {
            let mut evals = r_alpha_poly.evaluate_over_domain_by_ref(mul_domain);
            cfg_iter_mut!(evals.evals)
                .zip(&summed_z_m_evals.evals)
                .for_each(|(r_a, z_m)| *r_a *= z_m);
            evals.interpolate()
        };
        let tz_term_poly = {
            let mut evals = t_poly.evaluate_over_domain_by_ref(mul_domain);
            cfg_iter_mut!(evals.evals)
                .zip(&z_poly_evals.evals)
                .for_each(|(t, z)| *t *= z);
            -evals.interpolate()
        };
        let mul_term_poly = {
            let mut evals = z_a_times_z_b.evaluate_over_domain_by_ref(mul_domain);
            cfg_iter_mut!(evals.evals)
                .zip(&z_c_evals_mul.evals)
                .for_each(|(zab, zc)| *zab = eta_c * (*zab - zc));
            evals.interpolate()
        };

        cfg_iter_mut!(r_alpha_evals.evals)
            .zip(&summed_z_m_evals.evals)
            .zip(&z_poly_evals.evals)
            .zip(&t_poly_m_evals.evals)
            .zip(&z_a_z_b_evals.evals)
            .zip(&z_c_evals_mul.evals)
            .for_each(|(((((r_a, z_m), &z), t), z_a_z_b), z_c)| {
                *r_a *= z_m; // r_α · (η_A·z_A + η_B·z_B)
                *r_a -= z * t; // - t · z
                *r_a += eta_c * (*z_a_z_b - z_c); // + η_C·(z_A·z_B - z_C)
            });
        let rhs = r_alpha_evals.interpolate();
        // η·(ρ_t(X) - σ)·v_H(X): accounts for the t - Φ(α,·) zero-check folded into h_1.
        // t(X) - Φ(α,X) = (ρ_t(X) - σ)·v_H(X), so the verifier checks η·(t(β) - γ) = η·(ρ_t(β) - σ)·v_H(β).
        let rho_t_minus_sigma = &rho_t - &DensePolynomial::from_coefficients_slice(&[sigma]);
        let eta_t_minus_phi =
            &(&DensePolynomial::from_coefficients_slice(&[eta]) * &rho_t_minus_sigma) * &v_H_poly;
        let q_1 = &(s_1.polynomial() + &rhs) + &eta_t_minus_phi;
        end_timer!(q_1_time);

        let sumcheck_time = start_timer!(|| "Compute sumcheck h and g polys");
        let (h_1, x_g_1) = q_1.divide_by_vanishing_poly(domain_h).unwrap();
        let g_1 = DensePolynomial::from_coefficients_slice(&x_g_1.coeffs[1..]);
        end_timer!(sumcheck_time);

        let msg = ProverMsg::EmptyMessage;

        assert!(g_1.degree() <= domain_h.size() - 2);
        assert!(h_1.degree() <= 2 * domain_h.size() + zk_bound - 1);

        let oracles = ProverSecondOracles {
            t: LabeledPolynomial::new("t".into(), t_poly.clone(), None, None),
            g_1: LabeledPolynomial::new("g_1".into(), g_1, Some(domain_h.size() - 2), Some(1)),
            h_1: LabeledPolynomial::new("h_1".into(), h_1, None, None),
        };

        state.w_poly = None;
        state.verifier_first_msg = Some(*ver_message);
        state.t_poly = Some(t_poly);
        end_timer!(round_time);

        (msg, oracles, state)
    }

    /// Output the number of oracles sent by the prover in the second round.
    pub fn prover_num_second_round_oracles() -> usize {
        3
    }

    /// Output the degree bounds of oracles in the second round.
    pub fn prover_second_round_degree_bounds(
        info: &IndexInfo<F>,
    ) -> impl Iterator<Item = Option<usize>> {
        let h_domain_size =
            GeneralEvaluationDomain::<F>::compute_size_of_domain(info.num_constraints).unwrap();

        vec![None, Some(h_domain_size - 2), None].into_iter()
    }

    /// Output the third round message and the next state.
    /// Computes f(X) and γ = t(β) = Φ(α,β); sends γ as the prover message.
    pub fn prover_third_round<'a, R: RngCore>(
        ver_message: &VerifierSecondMsg<F>,
        mut prover_state: ProverState<'a, F>,
        _rng: &mut R,
    ) -> Result<(ProverMsg<F>, ProverThirdOracles<F>, ProverState<'a, F>), Error> {
        let round_time = start_timer!(|| "AHP::Prover::ThirdRound");

        let domain_h = prover_state.domain_h;
        let domain_k = prover_state.domain_k;
        let beta = ver_message.beta;
        prover_state.beta = Some(beta);

        let VerifierFirstMsg {
            eta,
            eta_a,
            eta_b,
            eta_c,
            alpha,
        } = prover_state.verifier_first_msg.expect(
            "ProverState should include verifier_first_msg when prover_third_round is called",
        );

        let t_poly = prover_state
            .t_poly
            .as_ref()
            .expect("ProverState should include t_poly when prover_third_round is called");
        let rho_t = prover_state
            .rho_t
            .as_ref()
            .expect("ProverState should include rho_t when prover_third_round is called");

        // γ = Φ(α,β) = t(β) - ρ_t(β)·v_H(β) + σ·v_H(β)
        // t(X) = Σ η_M M̂*(α,X) + ρ_t(X)·v_H(X)
        // Φ(α,X) = Σ η_M M̂*(α,X) + σ·v_H(X)
        // so Φ(α,β) = t(β) - ρ_t(β)·v_H(β) + σ·v_H(β)
        let sigma = prover_state
            .sigma
            .expect("ProverState should include sigma when prover_third_round is called");
        let v_H_at_beta = domain_h.evaluate_vanishing_polynomial(beta);
        let gamma_claim =
            t_poly.evaluate(&beta) - rho_t.evaluate(&beta) * v_H_at_beta + sigma * v_H_at_beta;
        prover_state.gamma_claim = Some(gamma_claim);

        let v_H_at_alpha = domain_h.evaluate_vanishing_polynomial(alpha);
        let v_H_alpha_v_H_beta = v_H_at_alpha * v_H_at_beta;
        let eta_a_vv = eta_a * v_H_alpha_v_H_beta;
        let eta_b_vv = eta_b * v_H_alpha_v_H_beta;

        let joint_arith = &prover_state.index.joint_arith;
        let (row_on_K, col_on_K) = (&joint_arith.evals_on_K.row, &joint_arith.evals_on_K.col);

        let f_evals_time = start_timer!(|| "Computing f evals on K");
        let mut inverses: Vec<_> = cfg_into_iter!(0..domain_k.size())
            .map(|i| (beta - row_on_K[i]) * (alpha - col_on_K[i]))
            .collect();
        ark_ff::batch_inversion(&mut inverses);

        let (val_a_on_K, val_b_on_K) =
            (&joint_arith.evals_on_K.val_a, &joint_arith.evals_on_K.val_b);
        // f only sums over M ∈ {A, B}; C is excluded from the inner sumcheck
        let f_evals_on_K: Vec<_> = cfg_into_iter!(0..(domain_k.size()))
            .map(|i| inverses[i] * (eta_a_vv * val_a_on_K[i] + eta_b_vv * val_b_on_K[i]))
            .collect();
        end_timer!(f_evals_time);

        let f_poly_time = start_timer!(|| "Computing f poly");
        let f_unblinded =
            EvaluationsOnDomain::from_vec_and_domain(f_evals_on_K, domain_k).interpolate();
        // Choose ρ_f so that (s_2·v_H(β) + f - γ/|K|) is divisible by X,
        // which is required for g_2 = (s_2·v_H(β) + f - γ/|K|) / X to be a polynomial.
        // Divisibility by X requires: s_2(0)·v_H(β) + f(0) - γ/|K| = 0
        //   s_2(0) = σ/|K|  (constant term of s_2)
        //   f(0) = f_unblinded(0) + ρ_f·v_K(0)
        //   γ/|K| = t(β)/|K|
        // Solving: ρ_f = (γ/|K| - s_2(0)·v_H(β) - f_unblinded(0)) / v_K(0)
        let k_size = domain_k.size_as_field_element();
        let v_K_poly: DensePolynomial<F> = domain_k.vanishing_polynomial().into();
        let v_K_at_0 = v_K_poly.evaluate(&F::zero());
        let s_2_poly = prover_state.s_2.as_ref().unwrap().polynomial().clone();
        let s_2_at_0 = s_2_poly.evaluate(&F::zero());
        let f_unblinded_at_0 = f_unblinded.evaluate(&F::zero());
        // Choose ρ_f so that: s_2(0)·v_H(β) + f(0) - γ/|K| = 0
        // where f(0) = f_unblinded(0) + ρ_f·v_K(0)
        let rho_f = (gamma_claim / k_size - s_2_at_0 * v_H_at_beta - f_unblinded_at_0) / v_K_at_0;
        let f = &f_unblinded + &(&DensePolynomial::from_coefficients_slice(&[rho_f]) * &v_K_poly);
        end_timer!(f_poly_time);

        let f_labeled = LabeledPolynomial::new("f".to_string(), f, None, Some(1));
        prover_state.f_poly = Some(f_labeled.clone());
        // Restore verifier_first_msg so round 4 can access it
        prover_state.verifier_first_msg = Some(VerifierFirstMsg {
            eta,
            eta_a,
            eta_b,
            eta_c,
            alpha,
        });

        let msg = ProverMsg::FieldElements(vec![gamma_claim]);
        let oracles = ProverThirdOracles { f: f_labeled };
        end_timer!(round_time);

        Ok((msg, oracles, prover_state))
    }

    /// Output the number of oracles sent by the prover in the third round.
    pub fn prover_num_third_round_oracles() -> usize {
        1
    }

    /// Output the degree bounds of oracles in the third round.
    pub fn prover_third_round_degree_bounds(
        _info: &IndexInfo<F>,
    ) -> impl Iterator<Item = Option<usize>> {
        vec![None].into_iter()
    }

    /// Output the fourth round message and the next state.
    /// Receives ζ from the verifier; computes g_2 and h_2.
    pub fn prover_fourth_round<'a, R: RngCore>(
        _ver_message: &VerifierThirdMsg<F>,
        prover_state: ProverState<'a, F>,
        _r: &mut R,
    ) -> Result<(ProverMsg<F>, ProverFourthOracles<F>), Error> {
        let round_time = start_timer!(|| "AHP::Prover::FourthRound");

        let ProverState {
            index,
            verifier_first_msg,
            domain_h,
            domain_k,
            s_2,
            f_poly,
            gamma_claim,
            beta,
            ..
        } = prover_state;

        let s_2 = s_2.expect("ProverState should include s_2 when prover_fourth_round is called");
        let f_labeled =
            f_poly.expect("ProverState should include f_poly when prover_fourth_round is called");
        let gamma = gamma_claim
            .expect("ProverState should include gamma_claim when prover_fourth_round is called");
        let beta =
            beta.expect("ProverState should include beta when prover_fourth_round is called");

        let VerifierFirstMsg {
            eta_a,
            eta_b,
            alpha,
            ..
        } = verifier_first_msg.expect(
            "ProverState should include verifier_first_msg when prover_fourth_round is called",
        );

        let k_size = domain_k.size_as_field_element();

        let v_H_at_alpha = domain_h.evaluate_vanishing_polynomial(alpha);
        let v_H_at_beta = domain_h.evaluate_vanishing_polynomial(beta);
        let v_H_alpha_v_H_beta = v_H_at_alpha * v_H_at_beta;
        let eta_a_vv = eta_a * v_H_alpha_v_H_beta;
        let eta_b_vv = eta_b * v_H_alpha_v_H_beta;

        let joint_arith = &index.joint_arith;

        // a(X) = v_H(α)·v_H(β)·(η_A·val_A(X) + η_B·val_B(X))  — C excluded
        let a_poly_time = start_timer!(|| "Computing a poly");
        let a_poly = {
            let a = joint_arith.val_a.coeffs();
            let b = joint_arith.val_b.coeffs();
            let coeffs: Vec<F> = cfg_iter!(a)
                .zip(b)
                .map(|(a, b)| eta_a_vv * a + eta_b_vv * b)
                .collect();
            DensePolynomial::from_coefficients_vec(coeffs)
        };
        end_timer!(a_poly_time);

        let b_poly_time = start_timer!(|| "Computing b poly");
        let alpha_beta = alpha * beta;
        // b(X) = αβ - α·row(X) - β·col(X) + row_col(X)
        // Use the blinded polynomials directly so the verifier's LC (which also
        // evaluates the blinded committed polynomials) is consistent.
        let b_poly = {
            let row_coeffs = joint_arith.row.coeffs();
            let col_coeffs = joint_arith.col.coeffs();
            let row_col_coeffs = joint_arith.row_col.coeffs();
            let len = row_coeffs
                .len()
                .max(col_coeffs.len())
                .max(row_col_coeffs.len())
                + 1;
            let mut coeffs = vec![F::zero(); len];
            coeffs[0] += alpha_beta;
            for (i, &r) in row_coeffs.iter().enumerate() {
                coeffs[i] -= alpha * r;
            }
            for (i, &c) in col_coeffs.iter().enumerate() {
                coeffs[i] -= beta * c;
            }
            for (i, &rc) in row_col_coeffs.iter().enumerate() {
                coeffs[i] += rc;
            }
            DensePolynomial::from_coefficients_vec(coeffs)
        };
        end_timer!(b_poly_time);

        let f = f_labeled.polynomial();
        let s_2_poly = s_2.polynomial();
        let v_H_at_beta = domain_h.evaluate_vanishing_polynomial(beta);

        // g_2 = (s_2(X)·v_H(β) + f(X) - γ/|K|) / X
        // By construction of ρ_f in round 3, the constant term vanishes so X divides this.
        let g_2 = {
            let s_2_scaled = &DensePolynomial::from_coefficients_slice(&[v_H_at_beta]) * s_2_poly;
            let mut numerator = &s_2_scaled + f;
            numerator.coeffs[0] -= gamma / k_size;
            // Divide by X: constant term should be 0; shift coefficients down
            debug_assert!(
                numerator.coeffs[0].is_zero(),
                "constant term of g_2 numerator must be zero"
            );
            DensePolynomial::from_coefficients_slice(&numerator.coeffs[1..])
        };

        let h_2_poly_time = start_timer!(|| "Computing h_2 poly");
        // h_2·v_K = a - b·f  (the verifier check at ζ adds the ζ-correction term)
        let h_2 = (&a_poly - &(&b_poly * f))
            .divide_by_vanishing_poly(domain_k)
            .unwrap()
            .0;
        end_timer!(h_2_poly_time);

        let msg = ProverMsg::EmptyMessage;

        assert!(g_2.degree() <= domain_k.size() - 1);
        let oracles = ProverFourthOracles {
            g_2: LabeledPolynomial::new("g_2".to_string(), g_2, Some(domain_k.size() - 1), None),
            h_2: LabeledPolynomial::new("h_2".to_string(), h_2, None, None),
        };
        end_timer!(round_time);

        Ok((msg, oracles))
    }

    /// Output the number of oracles sent by the prover in the fourth round.
    pub fn prover_num_fourth_round_oracles() -> usize {
        2
    }

    /// Output the degree bounds of oracles in the fourth round.
    pub fn prover_fourth_round_degree_bounds(
        info: &IndexInfo<F>,
    ) -> impl Iterator<Item = Option<usize>> {
        let num_non_zero = info.num_non_zero;
        let k_size = GeneralEvaluationDomain::<F>::compute_size_of_domain(num_non_zero).unwrap();
        // g_2 degree bound is |K|-1 (f is blinded with v_K, so g_2 can reach degree |K|-1)
        vec![Some(k_size - 1), None].into_iter()
    }
}
