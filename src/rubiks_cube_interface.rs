use crate::rubiks_solver::RubiksSolver;
use crate::rubiks::RubiksCube;
use crate::rubiks::CubeMove;
use crate::rubiks_solver::Trajectory;
use rand::Rng;
use log::{info};

pub struct RubiksCubeModelInterface {
    rubiks_cube: RubiksCube,
    solver: RubiksSolver,
    num_trajectories: u32,
    trajectory_depth: u32,
    num_layers: u32,
    hidden_layer_dimension: u32,
    num_epochs: u32,
    learning_rate: f64,
    num_tests: u32
}

impl RubiksCubeModelInterface {
    pub fn new() -> Self {
        let num_trajectories = 400;
        let trajectory_depth = 2;
        let num_layers: u32 = 5;
        let hidden_layer_dimension: u32 = 2000;
        let num_epochs: u32 = 50;
        let learning_rate: f64 = 1e-4;
        RubiksCubeModelInterface {
            rubiks_cube: RubiksCube::new(),
            solver: RubiksSolver::new(num_layers,hidden_layer_dimension,
                num_epochs,num_trajectories,trajectory_depth,
                learning_rate),
            num_trajectories: num_trajectories,
            trajectory_depth: trajectory_depth,
            num_layers: num_layers,
            hidden_layer_dimension: hidden_layer_dimension,
            num_epochs: num_epochs,
            learning_rate: learning_rate,
            num_tests: 10
        }
    }

    fn random_sample_move() -> CubeMove {
        let mut rng = rand::rng();
        let n: u8 = rng.random_range(0..=11);
        let mv = RubiksSolver::index_cube_move(n as u32);
        mv.expect("Expected move")
    }

    fn randomly_scramble_cube(cube: &RubiksCube,
        num_starting_points: u32,turns: u32, same_end: bool) -> 
        (Vec<RubiksCube>,Vec<Vec<CubeMove>>) {
        let (mut cubes,mut moves_l) = (Vec::new(),Vec::new());
        let _num_starting_points = num_starting_points;
        let mut fixed_moves: Vec<CubeMove> = Vec::new();
        for i in 0..turns {
            fixed_moves.push(
                Self::random_sample_move()
                // CubeMove::FPlus
                // CubeMove::FMinus
                // CubeMove::UPlus
                // CubeMove::UMinus
                // CubeMove::DMinus
                // CubeMove::DPlus
                // CubeMove::LMinus
                // CubeMove::LPlus
                // CubeMove::RMinus
                // CubeMove::RPlus
            );
        }

        for i in 0.._num_starting_points {
            let mut cube = cube.clone();
            let mut moves = Vec::new();

            for i in 0..turns {
                let mv = if !same_end {
                    Self::random_sample_move()
                } else {
                    fixed_moves.get(i as usize).expect("Expected move index").clone()
                };
                moves.push(mv.clone());
                cube.update_rep(mv.clone());
            }
            cubes.push(cube);
            moves_l.push(moves);
        }
        (cubes,moves_l)
    }

    pub fn test_logits(&self) {
        //Test logits: 
        let logits = self.solver.generate_move_logits(&self.rubiks_cube);
        println!("Got the cubemove logits from trained policy: {:?}",logits.size());
        println!("Printing logits:");
        logits.print();
        println!("Sum of logits: {:?}",logits.sum(tch::Kind::Float));
    }

    pub fn test_representation(&self) {
        let mut cube = RubiksCube::new();
        let mut rep = RubiksSolver::gen_input_representation(&cube);
        println!("Solved cube presentation is: {:?}",rep.size());
        rep.print();
        cube.update_rep(CubeMove::BPlus);
        rep = RubiksSolver::gen_input_representation(&cube);
        println!("BPlus cube presentation is: {:?}",rep.size());
        rep.print();
    }

    pub fn test_reward_alloc(&self) {
        // Test reward allocation: 
        let mut new_c = self.rubiks_cube.apply_move(CubeMove::BMinus);
        new_c.update_rep(CubeMove::FMinus);
        let r = RubiksSolver::get_reward(&new_c,&self.rubiks_cube);
        println!("Received reward: {:?}",r);
    }

    pub fn test_policy(&mut self) {
        // Display the trajectories: 
        let r_cubes_moves: (Vec<RubiksCube>, Vec<Vec<CubeMove>>) = 
            Self::randomly_scramble_cube(&mut self.rubiks_cube, 
                10,self.trajectory_depth,false);
            info!("The random scrambling applied these moves: {:?} for testing",
                r_cubes_moves.1);
        let mut mean_loss: f32 = 0.0;
        let mut mean_entropy: f32 = 0.0;
        let mut num_correct_moves: u32 = 0;

        info!("Running the test loop now: ");
        info!("Number of tests: {:?}",self.num_tests);
        for i in 0..self.num_tests as usize {
            let mut r_cube = 
                r_cubes_moves.0.get(i).expect("Expected cube").clone();
            let r_move = r_cubes_moves.1.get(i).expect("Expected move");
            info!("Scrambled cube used for test {:?} used this moves: {:?}",i,r_move);

            for j in 0..self.trajectory_depth {
                let logits = self.solver.generate_move_logits(&r_cube);
                let mv = RubiksSolver::sample_action(&logits);
                let mvc = 
                    r_move.get((self.trajectory_depth - j - 1) as usize).expect("Expected move to be available");
                r_cube = r_cube.apply_move(mv.clone());
                let compl = CubeMove::are_cube_moves_complementary(&mv,&mvc);
                info!("Got the cubemove from policy: {:?} for turn {:?}",mv,j);
                info!("The above is complementary or not: {:?}",compl);
                num_correct_moves += if compl {
                    1
                } else {
                    0
                };

                let policy_entropy = - (&logits.log() * &logits).
                        sum_dim_intlist(1,false,tch::Kind::Float).
                        to_device(tch::Device::Cpu);
                let policy_entropy_f: f32 = policy_entropy.try_into().
                    expect("Failed to get entropy off the tensor");
                mean_entropy += policy_entropy_f;
                //Calculate test reward here, calculate mean entropy of test moves: 
                info!("The cross entropy for turn {:?} {:?}",
                        j,
                        policy_entropy_f);
                // info!("Printing logits: {:?} {:?}",logits.size(),logits.to_device(tch::Device::Cpu));
                // info!("Sum of logits: {:?}",logits.sum(tch::Kind::Float));
            }
        }
        info!("Mean loss for tests {:?}",mean_loss / (self.num_tests as f32));
        info!("Mean entropy for tests {:?}",mean_entropy / ((self.num_tests * self.trajectory_depth) as f32));
        info!("Mean correct moves for {:?}",num_correct_moves as f32 / self.num_tests as f32);
    } 

    pub fn train_policy(&mut self) {
        info!("Using the following hyperparameters:");
        info!("Hidden layer dimensions: {:?}",self.hidden_layer_dimension);
        info!("Number of layers: {:?}",self.num_layers);
        info!("Number of trajectories: {:?}",self.num_trajectories);
        info!("Trajectory depth: {:?}",self.trajectory_depth);
        info!("Number of epochs: {:?}",self.num_epochs);
        info!("Learning rate: {:?}",self.learning_rate);
        info!("Starting policy training...");

        for i in 0..self.solver.num_epochs {
            info!("Training epoch {}",i);
            //Start with a different starting point for each epoch: 
            let r_cubes_moves = 
                Self::randomly_scramble_cube(
                    &mut self.rubiks_cube, 
                    self.num_trajectories,self.trajectory_depth,false);
            println!("The random scrambling applied these moves: {:?} for training epoch {:?}",
                r_cubes_moves.1,i);

            let trajs: Vec<Trajectory> = 
                self.solver.gen_trajectories(r_cubes_moves.0.clone());
            // println!("Trajectories: {:?}",trajs);
            self.solver.train_an_epoch(trajs);
        }
    }
}