use ark_ff::Field;
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError},
};
use ark_std::marker::PhantomData;

#[derive(Copy, Clone)]
struct Circuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
    num_constraints: usize,
    num_variables: usize,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for Circuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a.mul_assign(&b);
            Ok(a)
        })?;
        let d = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a.mul_assign(&b);
            a.mul_assign(&b);
            Ok(a)
        })?;

        for _ in 0..(self.num_variables - 3) {
            let _ = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        }

        for _ in 0..(self.num_constraints - 1) {
            cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        }
        cs.enforce_constraint(lc!() + c, lc!() + b, lc!() + d)?;

        Ok(())
    }
}

#[derive(Clone)]
/// Define a constraint system that would trigger outlining.
struct OutlineTestCircuit<F: Field> {
    field_phantom: PhantomData<F>,
}

impl<F: Field> ConstraintSynthesizer<F> for OutlineTestCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // This program checks if the input elements are between 0 and 9.
        //
        // Note that this constraint system is neither the most intuitive way nor
        // the most efficient way for such a task. It is for testing purposes,
        // as we want to trigger the outlining.
        //
        let mut inputs = Vec::new();
        for i in 0..5 {
            inputs.push(cs.new_input_variable(|| Ok(F::from(i as u128)))?);
        }

        for i in 0..5 {
            let mut total_count_for_this_input = cs.new_lc(lc!()).unwrap();

            for bucket in 0..10 {
                let count_increment_for_this_bucket =
                    cs.new_witness_variable(|| Ok(F::from(i == bucket)))?;

                total_count_for_this_input = cs
                    .new_lc(
                        lc!()
                            + (F::one(), total_count_for_this_input)
                            + (F::one(), count_increment_for_this_bucket.clone()),
                    )
                    .unwrap();

                // Only when `input[i]` equals `bucket` can `count_increment_for_this_bucket` be nonzero.
                //
                // A malicious prover can make `count_increment_for_this_bucket` neither 0 nor 1.
                // But the constraint on `total_count_for_this_input` will reject such case.
                //
                // At a high level, only one of the `count_increment_for_this_bucket` among all the buckets
                // could be nonzero, which equals `total_count_for_this_input`. Thus, by checking whether
                // `total_count_for_this_input` is 1, we know this input number is in the range.
                //
                cs.enforce_constraint(
                    lc!() + (F::one(), inputs[i].clone())
                        - (F::from(bucket as u128), ark_relations::r1cs::Variable::One),
                    lc!() + (F::one(), count_increment_for_this_bucket),
                    lc!(),
                )?;
            }

            // Enforce `total_count_for_this_input` to be one.
            cs.enforce_constraint(
                lc!(),
                lc!(),
                lc!() + (F::one(), total_count_for_this_input.clone())
                    - (F::one(), ark_relations::r1cs::Variable::One),
            )?;
        }

        Ok(())
    }
}

/// Fibonacci circuit: given public inputs f_0, f_1, proves knowledge of a Fibonacci
/// chain of `num_steps` steps, with the last two values as public outputs.
///
/// H layout: x = (1, f_0, f_1), w_internal = (f_2, ..., f_{n-1}), y = (f_n, f_{n+1}).
/// The circuit adds enough dummy constraints so that num_constraints >= num_variables
/// **before** squaring, ensuring the squaring step adds dummy *constraints* (not witnesses)
/// and the output variables remain last in the witness assignment.
#[derive(Clone)]
struct FibonacciCircuit<F: Field> {
    /// The starting values (None in setup mode).
    f0: Option<F>,
    f1: Option<F>,
    /// Number of Fibonacci steps (chain has num_steps+2 values: f_0..f_{num_steps+1}).
    num_steps: usize,
}

impl<F: Field> ConstraintSynthesizer<F> for FibonacciCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let n = self.num_steps;
        assert!(n >= 2, "need at least 2 steps so there are 2 output values");

        // Compute the full Fibonacci chain if values are provided.
        let chain: Option<Vec<F>> = self.f0.zip(self.f1).map(|(f0, f1)| {
            let mut v = vec![f0, f1];
            for i in 2..(n + 2) {
                v.push(v[i - 1] + v[i - 2]);
            }
            v
        });

        let val = |i: usize| -> Result<F, SynthesisError> {
            chain.as_ref().map(|v| v[i]).ok_or(SynthesisError::AssignmentMissing)
        };

        // Public inputs: f_0, f_1  (after the leading 1 added by formatting)
        let vars_x: Vec<_> = (0..2)
            .map(|i| cs.new_input_variable(|| val(i)))
            .collect::<Result<_, _>>()?;

        // Internal witnesses: f_2 ... f_{n-1}
        let mut vars_w: Vec<_> = (2..n)
            .map(|i| cs.new_witness_variable(|| val(i)))
            .collect::<Result<_, _>>()?;

        // Public outputs — declared last so they sit at the end of witness_assignment.
        // We add dummy constraints below to ensure squaring never appends more witnesses.
        let vars_y: Vec<_> = (n..n + 2)
            .map(|i| cs.new_witness_variable(|| val(i)))
            .collect::<Result<_, _>>()?;

        // All vars in chain order for the Fibonacci constraints.
        let mut all_vars = vars_x;
        all_vars.append(&mut vars_w);
        all_vars.extend_from_slice(&vars_y);

        // Enforce f_i + f_{i+1} = f_{i+2}  (addition as multiplication by 1)
        for i in 0..n {
            cs.enforce_constraint(
                lc!() + all_vars[i].clone() + all_vars[i + 1].clone(),
                lc!() + (F::one(), ark_relations::r1cs::Variable::One),
                lc!() + all_vars[i + 2].clone(),
            )?;
        }

        // Add dummy constraints (0*0=0) until num_constraints >= num_variables,
        // so that make_matrices_square adds more constraints (not witnesses) and
        // the output variables remain the last entries in witness_assignment.
        let num_vars = cs.num_instance_variables() + cs.num_witness_variables();
        let num_cons = cs.num_constraints();
        if num_cons < num_vars {
            for _ in 0..(num_vars - num_cons) {
                cs.enforce_constraint(lc!(), lc!(), lc!())?;
            }
        }

        Ok(())
    }
}

mod marlin {
    use super::*;
    use crate::{Marlin, SimpleHashFiatShamirRng};

    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ff::UniformRand;
    use ark_poly::univariate::DensePolynomial;
    use ark_poly_commit::marlin_pc::MarlinKZG10;
    use ark_std::ops::MulAssign;
    use blake2::Blake2s;
    use rand_chacha::ChaChaRng;

    type MultiPC = MarlinKZG10<Bls12_381, DensePolynomial<Fr>>;
    type FS = SimpleHashFiatShamirRng<Blake2s, ChaChaRng>;
    type MarlinInst = Marlin<Fr, MultiPC, FS>;

    fn test_circuit(num_constraints: usize, num_variables: usize) {
        let rng = &mut ark_std::test_rng();

        let universal_srs = MarlinInst::universal_setup(100, 25, 300, rng).unwrap();

        for _ in 0..100 {
            let a = Fr::rand(rng);
            let b = Fr::rand(rng);
            let mut c = a;
            c.mul_assign(&b);
            let mut d = c;
            d.mul_assign(&b);

            let circ = Circuit {
                a: Some(a),
                b: Some(b),
                num_constraints,
                num_variables,
            };

            let (index_pk, index_vk) = MarlinInst::index(&universal_srs, circ.clone(), 0, rng).unwrap();
            println!("Called index");

            let proof = MarlinInst::prove(&index_pk, circ, rng).unwrap();
            println!("Called prover");

            assert!(MarlinInst::verify(&index_vk, &[c, d], &[], &proof, rng).unwrap());
            println!("Called verifier");
            println!("\nShould not verify (i.e. verifier messages should print below):");
            assert!(!MarlinInst::verify(&index_vk, &[a, a], &[], &proof, rng).unwrap());
        }
    }

    #[test]
    fn prove_and_verify_with_tall_matrix_big() {
        let num_constraints = 100;
        let num_variables = 25;

        test_circuit(num_constraints, num_variables);
    }

    #[test]
    fn prove_and_verify_with_tall_matrix_small() {
        let num_constraints = 26;
        let num_variables = 25;

        test_circuit(num_constraints, num_variables);
    }

    #[test]
    fn prove_and_verify_with_squat_matrix_big() {
        let num_constraints = 25;
        let num_variables = 100;

        test_circuit(num_constraints, num_variables);
    }

    #[test]
    fn prove_and_verify_with_squat_matrix_small() {
        let num_constraints = 25;
        let num_variables = 26;

        test_circuit(num_constraints, num_variables);
    }

    #[test]
    fn prove_and_verify_with_square_matrix() {
        let num_constraints = 25;
        let num_variables = 25;

        test_circuit(num_constraints, num_variables);
    }

    #[test]
    fn prove_and_verify_fibonacci() {
        let rng = &mut ark_std::test_rng();
        // 10 steps: chain f_0..f_11, inputs f_0,f_1, outputs f_10,f_11
        let num_steps = 10;

        let universal_srs = MarlinInst::universal_setup(200, 200, 200, rng).unwrap();

        let f0 = Fr::from(1u64);
        let f1 = Fr::from(1u64);
        let mut chain = vec![f0, f1];
        for i in 2..(num_steps + 2) {
            chain.push(chain[i - 1] + chain[i - 2]);
        }
        let public_output = vec![chain[num_steps], chain[num_steps + 1]];

        let circ = FibonacciCircuit { f0: Some(f0), f1: Some(f1), num_steps };

        let (index_pk, index_vk) =
            MarlinInst::index(&universal_srs, circ.clone(), 2, rng).unwrap();
        println!("Fibonacci: called index");

        let proof = MarlinInst::prove(&index_pk, circ, rng).unwrap();
        println!("Fibonacci: called prover");

        assert!(MarlinInst::verify(&index_vk, &[f0, f1], &public_output, &proof, rng).unwrap());
        println!("Fibonacci: verified correctly");

        // Wrong output should not verify
        let wrong_output = vec![chain[num_steps] + Fr::from(1u64), chain[num_steps + 1]];
        assert!(!MarlinInst::verify(&index_vk, &[f0, f1], &wrong_output, &proof, rng).unwrap());
        println!("Fibonacci: correctly rejected wrong output");
    }

    #[test]
    /// Test on a constraint system that will trigger outlining.
    fn prove_and_test_outlining() {
        let rng = &mut ark_std::test_rng();

        let universal_srs = MarlinInst::universal_setup(150, 150, 150, rng).unwrap();

        let circ = OutlineTestCircuit {
            field_phantom: PhantomData,
        };

        let (index_pk, index_vk) = MarlinInst::index(&universal_srs, circ.clone(), 0, rng).unwrap();
        println!("Called index");

        let proof = MarlinInst::prove(&index_pk, circ, rng).unwrap();
        println!("Called prover");

        let mut inputs = Vec::new();
        for i in 0..5 {
            inputs.push(Fr::from(i as u128));
        }

        assert!(MarlinInst::verify(&index_vk, &inputs, &[], &proof, rng).unwrap());
        println!("Called verifier");
    }
}
