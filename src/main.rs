use crate::{rubiks_cube_interface::RubiksCubeModelInterface};
use simplelog::*;
use std::{fs::File, str::FromStr};
use log::{info};
use chrono::{DateTime, Utc};

pub mod rubiks;
pub mod rubiks_solver;
pub mod rubiks_cube_interface;

fn main() {
    // let log_file = 
    //    Utc::now().format("%Y-%m-%d %H:%M:%S") + "run";
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Warn, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Info, Config::default(), File::create("run_7.log").unwrap()),
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
