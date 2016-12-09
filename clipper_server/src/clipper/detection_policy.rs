extern crate rand;
use rand::{Rng, SeedableRng, StdRng};
use server::{Input, InputType, Output, RetrainTuple};
use cmt::{InputTable, RedisInputTable, UpdateTable, RedisUpdateTable, Preds};
use ml::{linalg, linear};
use std::ptr;

pub trait DetectionPolicy {
    /// Each detection policy must provide a name.
    fn get_name() -> &'static str;

    /// Returns true if this detection policy accepts the provided input type.
    /// Used to check for valid configurations at runtime.
    fn accepts_input_type(input_type: &InputType) -> bool;

    /// Determine whether or not a significant shift has occurred
    /// and collect together the necessary parameters and data for retraining
    /// including data points and weights, then prompt retraining
    fn evaluate(uid: u32, input_table: &RedisInputTable,
                update_table: &RedisUpdateTable) -> (bool, Vec<RetrainTuple>);
}

pub struct LogisticRegressionDetection {}

impl LogisticRegressionDetection {
    fn select_training(uid: u32, input_table: &RedisInputTable,
                       update_table: &RedisUpdateTable) -> (Vec<Vec<f64>>,
                                                            Vec<f64>, Vec<(Input, Output)>) {
        // All of training examples and recent window of test inputs
        let train_inputs: Vec<(Input, Output)> = update_table.get_updates(uid, -1).unwrap();
        let test_inputs: Vec<(Input, Preds, Output)> = input_table.get_inputs(uid, -1).unwrap();
        let mut train_input_return: Vec<(Input, Output)> = Vec::new();
        println!("Number of train inputs: {}", train_inputs.len());
        println!("Number of test inputs: {}", test_inputs.len());
        let mut logreg_inputs: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        for (input, output) in train_inputs {
            train_input_return.push((input.clone(), output));
            let result: Vec<f64> = match input {
                Input::Floats {ref f, length: _} => f.clone(),
                _ => panic!("evaluate received a type other than Input::Floats!"),
            };
            logreg_inputs.push(result);
            labels.push(0.0);
        }
        // train_inputs = update_table.get_updates(uid, -1).unwrap();

        for (input, _, output) in test_inputs {
            train_input_return.push((input.clone(), output));
            let result: Vec<f64> = match input {
                Input::Floats {ref f, length: _} => f.clone(),
                _ => panic!("evaluate received a type other than Input::Floats!"),
            };
            logreg_inputs.push(result);
            labels.push(1.0);
        }

        // Shuffle the data
        let rand_seed: &[_] = &[1, 2, 3, 4];
        let mut seed: StdRng = SeedableRng::from_seed(rand_seed);
        let mut seed2: StdRng = SeedableRng::from_seed(rand_seed);
        let mut seed3: StdRng = SeedableRng::from_seed(rand_seed);
        seed.shuffle(&mut logreg_inputs);
        seed2.shuffle(&mut labels);
        seed3.shuffle(&mut train_input_return);

        (logreg_inputs, labels, train_input_return)
    }
}

impl DetectionPolicy for LogisticRegressionDetection {
    fn get_name() -> &'static str {
        "logistic_regression_detection_policy"
    }

    fn accepts_input_type(input_type: &InputType) -> bool {
        match input_type {
            // InputType::Integer(_) => true,
            &InputType::Float(_) => true,
            _ => false
        }
    }

    #[allow(unused_variables)]
    fn evaluate(uid: u32, input_table: &RedisInputTable,
                update_table: &RedisUpdateTable) -> (bool, Vec<RetrainTuple>) {
        let (logreg_inputs, labels, train_inputs) =
            LogisticRegressionDetection::select_training(uid, input_table, update_table);
        if logreg_inputs.len() == 0 {
            return (false, Vec::new())
        }

        // Construct params struct
        // let weight_labels: Vec<i32> = vec![0, 1];
        // let weight_label: Vec<f64> = vec![1.0, test_importance];
        let params = linear::Struct_parameter {
            solver_type: linear::L2R_LR,
            eps: 0.0001,
            C: 1.0f64,
            nr_weight: 0,
            weight_label: ptr::null_mut(),
            weight: ptr::null_mut(),
            p: 0.1,
            init_sol: ptr::null_mut(),
        };

        let prob = linear::Problem::from_training_data(&logreg_inputs, &labels);
        let model = linear::train_logistic_regression(prob, params);
        let mut probs: Vec<f64> = Vec::new();
        let mut preds: Vec<f64> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut reweighted_train: Vec<RetrainTuple> = Vec::new();
        println!("model weight: {:?}", &model.w);
        for (input, label) in logreg_inputs.iter().zip(labels.iter()) {
            let res = model.logistic_regression_predict_proba(&input) - label;
            println!("x, conf, label: {:?}, {}, {}, {}", input.clone(),
                     model.logistic_regression_predict(&input),
                     linalg::dot(&model.w, &input), label);
            probs.push(res.powi(2));
            weights.push(model.logistic_regression_reweighting(&input));
            if model.logistic_regression_predict(&input) == *label {
                preds.push(1.0);
            } else {
                preds.push(0.0);
            }
        }

        let mut index: usize = 0;
        for (input, output) in train_inputs {
            if labels[index] == 0.0 {
                let weight = weights[index];
                println!("Reweighting: {:?}, {}, {}", input, output, weight);
                // if weight > 15.0 {
                //     weight = 3.2e6;
                // } else {
                //     weight = weight.powf(4.0);
                // }
                let r = RetrainTuple {
                    query: input,
                    label: output,
                    weight: weight,
                };
                reweighted_train.push(r);
            }
            index += 1;
        }
        let mut sum: f64 = probs.iter().sum();
        sum /= probs.len() as f64;
        let mut accuracy: f64 = preds.iter().sum();
        accuracy /= preds.len() as f64;
        println!("Accuracy: {}", accuracy);
        println!("Confidence: {}", sum.sqrt());
        // Set a hard threshold on prediction accuracy/confidence
        (sum.sqrt() < 0.48, reweighted_train)
    }
}
