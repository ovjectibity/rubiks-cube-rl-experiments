use crate::rubiks_cube_interface::RubiksCubeModelInterface;
use chrono::{DateTime, Utc};
use log::info;
use rand::Rng;
use simplelog::*;
use std::{fs::File, str::FromStr};

pub mod rubiks;
pub mod rubiks_cube_interface;
pub mod rubiks_solver;

fn main() {
    let identifier = ((rand::rng().random::<f32>() as f32) * 100.0);
    let log_file: String = format!("run_logs/run_{}.log", identifier,);
    let pol_file = format!("./run_logs/pol_{}.safetensors", identifier);
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Warn,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        TermLogger::new(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            Config::default(),
            File::create(log_file).unwrap(),
        ),
    ])
    .unwrap();
    //Start the training:
    info!("Is MPS available: {:?}", tch::utils::has_mps());
    let mut intf = RubiksCubeModelInterface::new(None);
    intf.train_policy();
    intf.test_policy();
    println!("Saving the trained policy to {}", pol_file);
    intf.save_policy(pol_file);
    // intf.test_representation();
    // intf.test_reward_alloc();
}
