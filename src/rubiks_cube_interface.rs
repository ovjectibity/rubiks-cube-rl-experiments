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
            solver: RubiksSolver::new(5,50,
                1,20,1)
        }
    }

    fn random_sample_move() -> CubeMove {
        let mut rng = rand::rng();
        let n: u8 = rng.random_range(0..=11);
        let mv = RubiksSolver::index_cube_move(n as u32);
        mv.expect("Expected move")
    }

    fn randomly_scramble_cube(cube: &mut RubiksCube,turns: u32) -> Vec<CubeMove> {
        let mut moves = Vec::new();
        for i in 0..turns {
            let mv = Self::random_sample_move();
            cube.apply_move(mv.clone());
            moves.push(mv);
        }
        moves
    }

    pub fn train_policy(&mut self) {
        //Test reward allocation: 
        // let mut new_c = self.rubiks_cube.apply_move(CubeMove::BMinus);
        // new_c.update_rep(CubeMove::BPlus);
        // let r = RubiksSolver::get_reward(&new_c,&self.rubiks_cube);

        println!("Initializing policy training...");
        println!("Applying scramling of length {}",1);
        //IMPROVEMENT: We're always starting from the same starting point, 
        // fix that for generalisation
        self.rubiks_cube.update_rep(CubeMove::BMinus);
        // let r_mvs = Self::randomly_scramble_cube(&mut self.rubiks_cube, 1);
        // println!("The random scrambling applied these moves: {:?}",r_mvs);

        for i in 0..self.solver.num_epochs {
            println!("Training {} epoch",i);
            let trajs: Vec<(Vec<CubeMove>, Vec<RubiksCube>, Vec<f32>)> = 
                self.solver.gen_trajectories(self.rubiks_cube.clone());
            // println!("Trajectories: {:?}",trajs);
            //TODO: Optimise this, forward pass in both gen_traj & training
            self.solver.train_an_epoch(trajs);
        }

        // Display the trajectories: 
        // let mv = [self.solver.generate_move(&self.rubiks_cube),
        //                         self.solver.generate_move(&self.rubiks_cube),
        //                         self.solver.generate_move(&self.rubiks_cube),
        //                         self.solver.generate_move(&self.rubiks_cube)];
        // let rcb2 = self.rubiks_cube.apply_move(mv.clone());
        // println!("Got the cubemove from trained policy: {:?} {:?} {:?} {:?}",mv[0],mv[1],mv[2],mv[3]);
        let logits = self.solver.generate_move_logits(&self.rubiks_cube);
        // println!("Got the cubemove logits from trained policy: {}",logits.values());
        println!("Printing logits:");
        logits.print();

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
}