

use server;
use digits;
use features;
use linear_models::{linalg, linear};

pub fn run(features: Vec<(String, SocketAddr)>,
       num_users: isize,
       num_train_examples: usize,
       num_test_examples: usize,
       mnist_path: String) {

    println!("starting digits");
    let all_test_data = digits::load_mnist_dense(mnist_path).unwrap();
    let norm_test_data = digits::normalize(&all_test_data);

    println!("Test data loaded: {} points", norm_test_data.ys.len());

    let tasks = digits::create_online_dataset(&norm_test_data,
                                              &norm_test_data,
                                              num_train_examples,
                                              0,
                                              num_test_examples,
                                              num_users);


    let (features, handles): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                              .map(|(n, a)| features::create_feature_worker(n, a))
                              .unzip();
}

/// Benchmark struct containing the trained model for
/// a task as well as some test data to evaluate the task on

struct TrainedTask {
    pub task_id: usize,
    pub pref: f64,
    pub test_x: Vec<Arc<Vec<f64>>>,
    pub test_y: Vec<f64>,
    model: linear::LogisticRegressionModel,
    /// anytime estimator for each feature in case we don't have it in time
    pub anytime_estimators: Arc<RwLock<HashMap<String, f64>>>
}

/// Wait until all features for all tasks have been computed asynchronously
fn get_all_train_features(tasks: Vec<DigitsTask>, feature_handles: &Vec<FeatureHandle>) {

    for t in &tasks {
        for x in t.offline_train_x {
            server::get_features(feature_handles, (&x).clone());
        }
    }
    println!("request all training features");
    loop {
        let sleep_secs = 5;
        println!("Sleeping {} seconds...", sleep_secs);
        thread::sleep(::std::time::Duration::new(sleep_secs, 0));
        for t in &tasks {
            for x in t.offline_train_x {


            }
        }


    }


}

fn pretrain_task_models(tasks: Vec<DigitsTask>, feature_handles: Vec<FeatureHandle>) 
    -> Vec<(DigitsTask, TaskModel)> {

    
    let mut trained_tasks = Vec::new();

    let tasks_with_models = tasks.into_iter().enumerate().map(|t| {
        let mut x_features: Vec<Arc<Vec<f64>>> = Vec::new();
        for x in t.offline_train_x {
            server::get_features(feature_handles, (&x).clone());
        }
        (t, TaskModel::train(i, t.pref, t.offline_train_x, t.offline_train_y))
    }).collect::<Vec<_>>();
    tasks_with_models
}

// #[allow(dead_code)]
#[derive(Debug)]
pub struct TaskModel {
    tid: usize, //model/task ID
    pub pref: f64,
    model: linear::LogisticRegressionModel
}

impl TaskModel {

    /// Constructs a new `TaskModel`
    /// `k` is the dimension of the model.
    /// `tid` is the task id assigned to this task (e.g. user ID).
    // pub fn new(k: usize, tid: usize) -> TaskModel {
    //     TaskModel {
    //         w: linalg::gen_normal_vec(k),
    //         tid: tid,
    //         k: k,
    //         model: None
    //     }
    // }

    pub fn get_id(&self) -> usize {
        self.tid
    }

    // pub fn get_wi(&self, i: usize) -> f64 {
    //     if i >= self.w.len() {
    //         0.0
    //     } else {
    //         self.w[i]
    //     }
    // }

    pub fn get_labels(&self) -> (f64, f64) {
        self.model.get_labels()
    }


    pub fn train(tid: usize, pref: f64, xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> TaskModel {
        let params = linear::Struct_parameter {
            solver_type: linear::L2R_LR,
            eps: 0.0001,
            C: 1.0f64,
            nr_weight: 0,
            weight_label: ptr::null_mut(),
            weight: ptr::null_mut(),
            p: 0.1,
            init_sol: ptr::null_mut()
        };
        let prob = linear::Problem::from_training_data(xs, ys);
        let model = linear::train_logistic_regression(prob, params);

        TaskModel {
            tid: tid,
            pref: pref,
            model: model
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        self.model.logistic_regression_predict(x)
    }
}