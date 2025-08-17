use crate::{rubiks_cube_interface::RubiksCubeModelInterface};

pub mod rubiks;
pub mod rubiks_solver;
pub mod rubiks_cube_interface;

fn main() {
    //Start the training: 
    let mut intf = RubiksCubeModelInterface::new();
    intf.train_policy();
    intf.test_policy();
    // intf.test_reward_alloc();
}
