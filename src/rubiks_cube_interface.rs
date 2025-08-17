use crate::rubiks_solver::RubiksSolver;
use crate::rubiks::RubiksCube;
use crate::rubiks::CubeMove;
use rand::Rng;

pub struct RubiksCubeModelInterface {
    rubiks_cube: RubiksCube,
    solver: RubiksSolver
}

impl RubiksCubeModelInterface {
    pub fn new() -> Self {
        RubiksCubeModelInterface {
            rubiks_cube: RubiksCube::new(),
            solver: RubiksSolver::new(5,100,
                200,100,1,
            1e-3)
        }
    }

    fn random_sample_move() -> CubeMove {
        let mut rng = rand::rng();
        let n: u8 = rng.random_range(0..=11);
        let mv = RubiksSolver::index_cube_move(n as u32);
        mv.expect("Expected move")
    }

    fn randomly_scramble_cube(cube: &RubiksCube,turns: u32) -> 
        Vec<(RubiksCube,CubeMove)> {
        let mut cubes_moves = Vec::new();
        for i in 0..turns {
            let mv = Self::random_sample_move();
            let cube = cube.apply_move(mv.clone());
            cubes_moves.push((cube,mv));
        }
        cubes_moves
    }

    pub fn test_logits(&self) {
        //Test logits: 
        let logits = self.solver.generate_move_logits(&self.rubiks_cube);
        println!("Got the cubemove logits from trained policy: {:?}",logits.size());
        println!("Printing logits:");
        logits.print();
        println!("Sum of logits: {:?}",logits.sum(tch::Kind::Float));
    }

    pub fn test_reward_alloc(&self) {
        // Test reward allocation: 
        let mut new_c = self.rubiks_cube.apply_move(CubeMove::BMinus);
        new_c.update_rep(CubeMove::BPlus);
        let r = RubiksSolver::get_reward(&new_c,&self.rubiks_cube);
        println!("Received reward: {:?}",r);
    }

    pub fn test_policy(&mut self) {
        // Display the trajectories: 
        //Scramble self cube: 
        let r_cubes_moves = Self::randomly_scramble_cube(&mut self.rubiks_cube, 1);
            let r_cubes_move = r_cubes_moves.get(0).
                expect("Expected cube & move at index 0");
            println!("The random scrambling applied these moves: {:?} for testing",
                r_cubes_move.1);
        let mv = [self.solver.generate_move(&r_cubes_move.0),
                                self.solver.generate_move(&r_cubes_move.0),
                                self.solver.generate_move(&r_cubes_move.0),
                                self.solver.generate_move(&r_cubes_move.0)];
        // let rcb2 = self.rubiks_cube.apply_move(mv.clone());
        println!("Got the cubemove from trained policy: {:?} {:?} {:?} {:?}",mv[0],mv[1],mv[2],mv[3]);
        let logits = self.solver.generate_move_logits(&r_cubes_move.0);
        // println!("Got the cubemove logits from trained policy: {}",logits.values());
        println!("Printing logits:");
        logits.print();
        println!("Sum of logits: {:?}",logits.sum(tch::Kind::Float));

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
                Self::randomly_scramble_cube(&mut self.rubiks_cube, 1);
            let r_cubes_move = r_cubes_moves.get(0).
                expect("Expected cube & move at index 0");
            println!("The random scrambling applied these moves: {:?} for training epoch {:?}",
                r_cubes_move.1,i);

            let trajs: Vec<(Vec<CubeMove>, Vec<RubiksCube>, Vec<f32>)> = 
                self.solver.gen_trajectories(r_cubes_move.0.clone());
            // println!("Trajectories: {:?}",trajs);
            //TODO: Optimise this, forward pass in both gen_traj & training
            self.solver.train_an_epoch(trajs);
        }
    }
}