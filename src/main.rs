use crate::{rubiks_cube_interface::RubiksCubeModelInterface};
use tch::Tensor;

pub mod rubiks;
pub mod rubiks_solver;
pub mod rubiks_cube_interface;

fn main() {
    //Start the training: 
    println!("Is MPS available: {:?}",tch::utils::has_mps());
    let mut intf = RubiksCubeModelInterface::new();
    intf.train_policy();
    intf.test_policy();
    // intf.test_representation();
    // intf.test_reward_alloc();
}
