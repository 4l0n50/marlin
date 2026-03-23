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

    // --- 1. Build the polynomial by interpolating the values [0..15] ---
    // The unique degree-15 polynomial p such that p(domain[i]) = i.
    let domain: GeneralEvaluationDomain<Fr> =
        GeneralEvaluationDomain::new(16).expect("domain of size 16 must exist");

    let evals: Vec<Fr> = (0u64..16).map(Fr::from).collect();
    let poly = EvaluationsOnDomain::from_vec_and_domain(evals, domain).interpolate();

    // domain.element(5) is the unique x in the domain with p(x) = 5
    let point = domain.element(5);
    assert_eq!(poly.evaluate(&point), Fr::from(5u64));
    println!("Polynomial degree: {}", poly.degree());

    // --- 2. Polynomial commitment setup ---
    let max_degree = poly.degree();
    let pp = PC::setup(max_degree, None, rng).unwrap();
    let (ck, vk) = PC::trim(&pp, max_degree, 1, None).unwrap();

    // --- 3. Commit ---
    let labeled_poly = LabeledPolynomial::new("p".to_string(), poly.clone(), None, Some(1));
    let (commitments, randomness) = PC::commit(&ck, vec![&labeled_poly], Some(rng)).unwrap();

    println!("Commitment: {:?}", commitments[0].commitment());

    // --- 4. Open at the point where p(x) = 5 ---
    let opening_challenge = Fr::rand(rng);
    let proof = PC::open(
        &ck,
        vec![&labeled_poly],
        &commitments,
        &point,
        opening_challenge,
        &randomness,
        Some(rng),
    )
    .unwrap();

    // --- 5. Verify ---
    let claimed_value = Fr::from(5u64);
    let is_valid = PC::check(
        &vk,
        &commitments,
        &point,
        vec![claimed_value],
        &proof,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    println!("Evaluation at domain[5]: p(x) = {}", claimed_value);
    println!("Proof valid: {}", is_valid);
    assert!(is_valid);

    // Sanity-check: a wrong claimed value should fail
    let is_invalid = PC::check(
        &vk,
        &commitments,
        &point,
        vec![Fr::from(99u64)],
        &proof,
        opening_challenge,
        Some(rng),
    )
    .unwrap();
    assert!(!is_invalid, "wrong value should not verify");
    println!("Wrong value (99) correctly rejected.");
}
