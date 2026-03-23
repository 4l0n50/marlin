// Example: commit to the polynomial interpolating [0, 1, 2, ..., 15] over a
// size-16 multiplicative domain, then prove and verify that p(x) = 5 at the
// domain point x = domain[5].
//
// Run with:
//   cargo run --example commit_and_open

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain, Polynomial,
};
use ark_poly_commit::{marlin_pc::MarlinKZG10, LabeledPolynomial, PolynomialCommitment};

type PC = MarlinKZG10<Bls12_381, DensePolynomial<Fr>>;

fn main() {
    let rng = &mut ark_std::test_rng();

    // --- 1. Build the polynomials R(X) and C(X) such that R(ω^i) = ∆^{r_i} and C(ω^i) = ∆^{c_i} with r_i < c_i ---

    let domain: GeneralEvaluationDomain<Fr> =
        GeneralEvaluationDomain::new(16).expect("domain of size 16 must exist");
    let twice_the_domain: GeneralEvaluationDomain<Fr> =
        GeneralEvaluationDomain::new(32).expect("Domain of size 32 exists");

    // r_evals = [∆⁰, ∆¹, ∆², ..., ∆¹⁵] the evaluations of R(X) over the domain
    let r_evals: Vec<Fr> = twice_the_domain.elements().take(16).collect();
    assert_eq!(r_evals.len(), 16);
    let r_poly = EvaluationsOnDomain::from_vec_and_domain(r_evals, domain).interpolate();
    println!("R(X) degree: {}", r_poly.degree());

    // c_evals = [∆¹, ∆², ∆³..., ∆¹⁶] the evaluations of C(X) over the domain
    let c_evals: Vec<Fr> = twice_the_domain.elements().skip(1).take(16).collect();
    assert_eq!(c_evals.len(), 16);
    let c_poly = EvaluationsOnDomain::from_vec_and_domain(c_evals, domain).interpolate();
    println!("C(X) degree: {}", c_poly.degree());

    // domain.element(5) is the unique x in the domain with R(x) = ∆⁵
    let w_5 = domain.element(5);
    assert_eq!(r_poly.evaluate(&w_5), twice_the_domain.element(5));
    // domain.element(5) is the unique x in the domain with C(x) = ∆⁶
    assert_eq!(c_poly.evaluate(&w_5), twice_the_domain.element(6));

    // domain.element(6) is the unique x in the domain with R(x) = ∆⁶
    let w_6 = domain.element(6);
    assert_eq!(r_poly.evaluate(&w_6), twice_the_domain.element(6));
    // domain.element(5) is the unique x in the domain with C(x) = ∆⁶
    assert_eq!(c_poly.evaluate(&w_6), twice_the_domain.element(7));

    // --- 2. Polynomial commitment setup ---
    let max_degree = r_poly.degree().max(c_poly.degree());
    let pp = PC::setup(max_degree, None, rng).unwrap();
    let (ck, vk) = PC::trim(&pp, max_degree, 1, None).unwrap();

    // --- 3. Commit ---
    let r_labeled_poly = LabeledPolynomial::new("R(X)".to_string(), r_poly.clone(), None, Some(1));
    let c_labeled_poly = LabeledPolynomial::new("C(X)".to_string(), c_poly.clone(), None, Some(1));
    let (commitments, randomness) =
        PC::commit(&ck, vec![&r_labeled_poly, &c_labeled_poly], Some(rng)).unwrap();

    println!("R(X) Commitment: {:?}", commitments[0].commitment());
    println!("C(X) Commitment: {:?}", commitments[1].commitment());

    // --- 4a. Open at the polys at ω⁵ ---
    let opening_challenge = Fr::rand(rng);
    let proof = PC::open(
        &ck,
        vec![&r_labeled_poly, &c_labeled_poly],
        &commitments,
        &w_5,
        opening_challenge,
        &randomness,
        Some(rng),
    )
    .unwrap();

    // --- 5a. Verify ---
    let claimed_values = [twice_the_domain.element(5), twice_the_domain.element(6)];
    let is_valid = PC::check(
        &vk,
        &commitments,
        &w_5,
        claimed_values,
        &proof,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    println!("Evaluation at ω⁵: R(ω⁵) = {}", claimed_values[0]);
    println!("Evaluation at ω⁵: C(ω⁵) = {}", claimed_values[1]);
    println!("Proof valid: {}", is_valid);
    assert!(is_valid);

    // --- 4b. Open at the polys at ω⁶ ---
    let opening_challenge = Fr::rand(rng);
    let proof = PC::open(
        &ck,
        vec![&r_labeled_poly, &c_labeled_poly],
        &commitments,
        &w_6,
        opening_challenge,
        &randomness,
        Some(rng),
    )
    .unwrap();

    // --- 5b. Verify ---
    let claimed_values = [twice_the_domain.element(6), twice_the_domain.element(7)];
    let is_valid = PC::check(
        &vk,
        &commitments,
        &w_6,
        claimed_values,
        &proof,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    println!("Evaluation at ω⁵: R(ω⁶) = {}", claimed_values[0]);
    println!("Evaluation at ω⁵: C(ω⁵) = {}", claimed_values[1]);
    println!("Proof valid: {}", is_valid);
    assert!(is_valid);

    // Sanity-check: a wrong claimed value should fail
    let is_invalid = PC::check(
        &vk,
        &commitments,
        &w_5,
        vec![Fr::from(99u64)],
        &proof,
        opening_challenge,
        Some(rng),
    )
    .unwrap();
    assert!(!is_invalid, "wrong value should not verify");
    println!("Wrong value (99) correctly rejected.");
}
