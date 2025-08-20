use crate::{rubiks_cube_interface::RubiksCubeModelInterface};
use simplelog::*;
use std::fs::File;
use log::{info};

pub mod rubiks;
pub mod rubiks_solver;
pub mod rubiks_cube_interface;

fn main() {
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Warn, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Info, Config::default(), File::create("run_1.log").unwrap()),
        ]
    ).unwrap();
    //Start the training: 
    info!("Is MPS available: {:?}",tch::utils::has_mps());
    let mut intf = RubiksCubeModelInterface::new();
    intf.train_policy();
    intf.test_policy();
    // intf.test_representation();
    // intf.test_reward_alloc();
}
