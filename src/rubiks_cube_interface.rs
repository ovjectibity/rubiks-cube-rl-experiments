use crate::rubiks_solver::RubiksSolver;
use crate::rubiks::RubiksCube;
use crate::rubiks::CubeMove;
use crate::rubiks_solver::Trajectory;
use rand::Rng;

pub struct RubiksCubeModelInterface {
    rubiks_cube: RubiksCube,
    solver: RubiksSolver,
    num_trajectories: u32,
    trajectory_depth: u32
}

impl RubiksCubeModelInterface {
    pub fn new() -> Self {
        let num_trajectories = 400;
        let trajectory_depth = 2;
        RubiksCubeModelInterface {
            rubiks_cube: RubiksCube::new(),
            solver: RubiksSolver::new(4,2500,
                50,num_trajectories,trajectory_depth,
            1e-4),
            num_trajectories: num_trajectories,
            trajectory_depth: trajectory_depth
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
            println!("The random scrambling applied these moves: {:?} for testing",
                r_cubes_moves.1);

        println!("Running the test loop now: ");
        for i in 0..10 {
            let mut r_cube = r_cubes_moves.0.get(i).expect("Expected cube").clone();
            let r_move = r_cubes_moves.1.get(i).expect("Expected move");
            println!("Scrambled cube used for test {:?} used this moves: {:?}",i,r_move);

            for j in 0..self.trajectory_depth {
                let mv = self.solver.generate_move(&r_cube);
                r_cube = r_cube.apply_move(mv.clone());
                println!("Got the cubemove from policy: {:?} for turn {:?}",mv,j);

                let logits = self.solver.generate_move_logits(&r_cube);
                let policy_entropy = - (&logits.log() * &logits).
                sum_dim_intlist(1,false,tch::Kind::Float);
                println!("The cross entropy for turn {:?} {:?}",j,policy_entropy.size());
                policy_entropy.print();
                println!("Printing logits: {:?}",logits.size());
                logits.print();
                println!("Sum of logits: {:?}",logits.sum(tch::Kind::Float));
            }
        }
    } 

    pub fn train_policy(&mut self) {
        println!("Initializing policy training...");

        for i in 0..self.solver.num_epochs {
            println!("Training {} epoch",i);
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