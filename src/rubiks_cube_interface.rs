use crate::rubiks_solver::RubiksSolver;
use crate::rubiks::RubiksCube;
use crate::rubiks::CubeMove;
use crate::rubiks_solver::Trajectory;
use rand::Rng;

pub struct RubiksCubeModelInterface {
    rubiks_cube: RubiksCube,
    solver: RubiksSolver,
    num_trajectories: u32
}

impl RubiksCubeModelInterface {
    pub fn new() -> Self {
        let num_trajectories = 400;
        RubiksCubeModelInterface {
            rubiks_cube: RubiksCube::new(),
            solver: RubiksSolver::new(4,2000,
                50,num_trajectories,1,
            1e-4),
            num_trajectories: num_trajectories
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
                // CubeMove::BPlus
            );
        }

        for i in 0.._num_starting_points {
            let mut cube = cube.clone();
            let mut moves = Vec::new();
            //Have the first move be deterministic: 
            // let first_mv = RubiksSolver::index_cube_move(i).expect("expected cube move");
            // cubes.push(cube.apply_move(first_mv.clone()));
            // moves.push(first_mv);

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
        //Scramble self cube: 
        let r_cubes_moves: (Vec<RubiksCube>, Vec<Vec<CubeMove>>) = 
            Self::randomly_scramble_cube(&mut self.rubiks_cube, 
                10,1,false);
            println!("The random scrambling applied these moves: {:?} for testing",
                r_cubes_moves.1);

        println!("Running the test loop now: ");
        for i in 0..10 {
            let r_cube = r_cubes_moves.0.get(i).expect("Expected cube");
            let r_move = r_cubes_moves.1.get(i).expect("Expected move").
                get(0).expect("Expected move");
            println!("Scrambled cube used for test used this move: {:?}",r_move);
            let mv = [self.solver.generate_move(r_cube),
                                    self.solver.generate_move(r_cube),
                                    self.solver.generate_move(r_cube),
                                    self.solver.generate_move(r_cube)];
            // let rcb2 = self.rubiks_cube.apply_move(mv.clone());
            println!("Got the cubemove from policy: {:?} {:?} {:?} {:?}",mv[0],mv[1],mv[2],mv[3]);
            let logits = self.solver.generate_move_logits(r_cube);
            let policy_entropy = - (&logits.log() * &logits).sum_dim_intlist(1,false,tch::Kind::Float);
            println!("The cross entropy for test cycle {:?} {:?}",i,policy_entropy.size());
            policy_entropy.print();
            println!("Printing logits: {:?}",logits.size());
            logits.print();
            println!("Sum of logits: {:?}",logits.sum(tch::Kind::Float));
        }

        // let mv2 = self.solver.generate_move(&rcb2);
        // let rcb2 = self.rubiks_cube.apply_move(mv.clone());
        // self.apply_move(CubeMove::UPlus,true);
        // self.apply_move(CubeMove::RPlus,true);
        // self.apply_move(CubeMove::DPlus,true);
        // self.apply_move(CubeMove::BPlus,true);
        // self.apply_move(CubeMove::LPlus);
        // self.apply_move(CubeMove::RPlus);
        // self.apply_move(CubeMove::RPlus);
        // self.apply_move(CubeMove::DPlus);
        // self.apply_move(CubeMove::FMinus,false);
        // self.apply_move(CubeMove::UMinus,false);
        // self.rubiks_cube.apply_move(CubeMove::FPlus);
        // self.rubiks_cube.apply_move(CubeMove::UPlus);
    } 

    pub fn train_policy(&mut self) {
        println!("Initializing policy training...");
        // println!("Applying scrambling of length {}",1);
        // self.rubiks_cube.update_rep(CubeMove::FPlus);

        for i in 0..self.solver.num_epochs {
            println!("Training {} epoch",i);
            //Start with a different starting point for each epoch: 
            let r_cubes_moves = 
                Self::randomly_scramble_cube(
                    &mut self.rubiks_cube, 
                    self.num_trajectories,1,false);
            println!("The random scrambling applied these moves: {:?} for training epoch {:?}",
                r_cubes_moves.1,i);

            let trajs: Vec<Trajectory> = 
                self.solver.gen_trajectories(r_cubes_moves.0.clone());
            // println!("Trajectories: {:?}",trajs);
            //TODO: Optimise this, forward pass in both gen_traj & training
            self.solver.train_an_epoch(trajs);
        }
    }
}