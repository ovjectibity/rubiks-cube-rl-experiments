use crate::rubiks::{CubeMove, Cubelet, FaceColor, RubiksCube, SlottedCubelet};
use log::info;
use rand::Rng;
use tch::{nn::{self, Module, OptimizerConfig}, no_grad, Tensor, TensorIndexer};

pub enum RLAlgorithm {
    PPO(PPOSolver),
    VPG(VPGSolver),
}

pub struct Trajectory {
    moves: Vec<CubeMove>,
    //Shape of logits: [num_possible_moves,trajectory_depth]
    logits: Tensor,
    rewards: Vec<f32>,
    cube_states: Tensor
}

impl Trajectory {
    fn new(cube_states: Tensor, moves: Vec<CubeMove>, logits: Tensor, rewards: Vec<f32>) -> Self {
        let mut traj = Trajectory {
            cube_states: cube_states,
            moves: moves,
            logits: logits,
            rewards: rewards,
        };
        traj.compute_rtg();
        traj
    }

    fn compute_rtg(&mut self) {
        let mut acc = 0.0;
        let mut rtgs: Vec<f32> = self
            .rewards
            .iter()
            .rev()
            .map(|reward| {
                acc += reward;
                acc
            })
            .collect();
        rtgs.reverse();
        self.rewards = rtgs;
    }

    fn compute_avg_entropy(&self) -> f32 {
        let entropy = -(&self.logits.log() * &self.logits)
            .sum_dim_intlist(1, false, tch::Kind::Float)
            .mean_dim(0, false, tch::Kind::Float);
        let x: f32 = entropy.try_into().expect("Failed to convert tensor to vec");
        x
    }
}

// struct PPOSolver {
//     policy: nn::Sequential,
//     store: nn::VarStore,
//     optim: nn::Optimizer,
// }

// impl PPOSolver {
//     fn new() -> Self {

//     }
// }

pub struct RubiksSolver {
    policy: nn::Sequential,
    trajectory_depth: u32,
    num_trajectories: u32,
    num_layers: u32,
    hidden_layer_dimension: u32,
    pub num_epochs: u32,
    store: nn::VarStore,
    optim: nn::Optimizer,
    algo: RLAlgorithm,
    advnet: Option<nn::Sequential>,
}

//Uses the policy gradient algorithm
impl RubiksSolver {
    pub fn new(
        num_layers: u32,
        hidden_layer_dimension: u32,
        num_epochs: u32,
        num_trajectories: u32,
        trajectory_depth: u32,
        learning_rate: f64,
        load_from_file: Option<String>,
    ) -> Self {
        let pols = Self::init_policy(
            load_from_file,
            hidden_layer_dimension as i64,
            num_layers as i64,
        );
        let mut optim = nn::Adam::default().build(&pols.1, learning_rate).unwrap();
        optim.set_weight_decay(1e-3);

        RubiksSolver {
            policy: pols.0,
            num_layers: num_layers,
            hidden_layer_dimension: hidden_layer_dimension,
            num_epochs: num_epochs,
            num_trajectories: num_trajectories,
            trajectory_depth: trajectory_depth,
            store: pols.1,
            optim: optim,
            algo: RLAlgorithm::VPG,
            advnet: None,
        }
    }

    pub fn save_policy(&self, path: String) {
        //Save trained policy to storage:
        self.store.save(path);
    }

    fn init_policy(
        load_from_file: Option<String>,
        hidden_layer_dimension: i64,
        num_layers: i64,
    ) -> (nn::Sequential, nn::VarStore) {
        let mut vs = nn::VarStore::new(tch::Device::Mps);
        let vs_p = vs.root();
        let mut y = nn::seq()
            .add(nn::linear(
                vs_p.clone(),
                324,
                hidden_layer_dimension,
                Default::default(),
            ))
            .add_fn(Tensor::relu);
        for i in 0..num_layers - 2 {
            y = y
                .add(nn::linear(
                    vs_p.clone(),
                    hidden_layer_dimension,
                    hidden_layer_dimension,
                    Default::default(),
                ))
                .add_fn(Tensor::relu);
        }
        y = y
            .add(nn::linear(
                vs_p.clone(),
                hidden_layer_dimension,
                13,
                Default::default(),
            ))
            .add_fn(Tensor::relu)
            .add_fn(|xs| xs.softmax(-1, tch::Kind::Float));
        if let Some(path) = load_from_file {
            vs.load(path);
        } else {
            //Initialise the tensors uniformly in the policy network:
            no_grad(|| {
                for (name, mut tensor) in vs.variables() {
                    if name.ends_with("weight") {
                        tensor.uniform_(-0.05, 0.05); // in-place
                    } else if name.ends_with("bias") {
                        tensor.zero_();
                    }
                }
            });
        }
        (y, vs)
    }

    pub fn generate_move_logits(&self, cube: &RubiksCube) -> Tensor {
        let input = Self::gen_input_representation(cube);
        // println!("Generated input tensor: {:?} {:?}",input.dim(),input);
        // println!("Policy tensor: {:?} {:?}",self.policy,&self.policy);
        let output = self.policy.forward(&input.transpose(0, 1));
        output
    }

    pub fn generate_move(&self, cube: &RubiksCube) -> CubeMove {
        let input = Self::gen_input_representation(cube);
        // println!("Generated input tensor: {:?} {:?}",input.dim(),input);
        // println!("Policy tensor: {:?} {:?}",self.policy,&self.policy);
        let output = self.policy.forward(&input.transpose(0, 1));
        // println!("Generated output tensor: {:?} {:?}",output.dim(),output);
        Self::sample_action(&output)
    }

    pub fn get_score(cube: &RubiksCube) -> f32 {
        let mut score: f32 = 0.0;
        let solved_cube = RubiksCube::new();
        let face_strings = Self::get_face_strings();
        let scm = cube.cube_slot_map.borrow();
        let solved_scm = solved_cube.cube_slot_map.borrow();
        for face_string in face_strings {
            let tar_cubelet = scm.get(&face_string).expect("Expected slotted cubelet");
            let sol_cubelet = solved_scm
                .get(&face_string)
                .expect("Expected slotted cubelet");
            match (tar_cubelet, sol_cubelet) {
                (SlottedCubelet::Corner(c1, i1), SlottedCubelet::Corner(c2, i2)) => {
                    // println!("Cubelet & color indices for the compared corner cubelets: {:?} {:?} {:?} {:?}",
                    //         i1,i2,c1.get_raw_color_indices(),c2.get_raw_color_indices());
                    if i1 == i2 && c1.get_raw_color_indices() == c2.get_raw_color_indices() {
                        score += 1.0;
                    }
                    // else if i1 == i2 {
                    //     score += 0.5;
                    // }
                }
                (SlottedCubelet::Edge(c1, i1), SlottedCubelet::Edge(c2, i2)) => {
                    // println!("Cubelet & color indices for the compared edge cubelets: {:?} {:?} {:?} {:?}",
                    //         i1,i2,c1.get_raw_color_indices(),c2.get_raw_color_indices());
                    if i1 == i2 && c1.get_raw_color_indices() == c2.get_raw_color_indices() {
                        score += 1.0;
                    }
                    // else if i1 == i2 {
                    //     score += 0.5;
                    // }
                }
                (SlottedCubelet::Center(i1), SlottedCubelet::Center(i2)) => {
                    //do nothing
                }
                _ => panic!("Unexpected cubelet combo"),
            }
        }
        // println!("Score is {}",score);
        // score.powi(2)
        (2.0 as f32).powf(score)
    }

    pub fn get_reward(cube_current: &RubiksCube, cube_previous: &RubiksCube) -> f32 {
        let new_score = Self::get_score(cube_current);
        let old_score = Self::get_score(cube_previous);
        // println!("Getting reward as {:?} {:?}",new_score,old_score);
        //BUG/ISSUE: Possible issue here, when a cube is scrambled further away the
        //the reward is being increased from the previous state:
        //For example no_op on the previous cube would get the reward as 0.
        new_score - old_score
    }

    pub fn sample_action(probs: &Tensor) -> CubeMove {
        assert!(probs.dim() == 2);
        // println!("Sampling action from probs: {:?}",probs.size());
        let index = Self::sample_categorical(&Vec::<f32>::try_from(probs.squeeze()).unwrap());
        Self::index_cube_move(index as u32).unwrap()
    }

    fn sample_categorical(probabilities: &[f32]) -> usize {
        let mut rng = rand::rng();
        let r: f32 = rng.random(); // random float in [0, 1)
        let mut cumulative = 0.0;

        for (i, &p) in probabilities.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i;
            }
        }
        // In case of rounding errors, return the last index
        probabilities.len() - 1
    }

    fn get_face_colors_corner<'a>(
        face_colors: (&'a FaceColor, &'a FaceColor, &'a FaceColor),
        color_indices: (u32, u32, u32),
    ) -> (&'a FaceColor, &'a FaceColor, &'a FaceColor) {
        match color_indices {
            (0, 1, 2) => (face_colors.0, face_colors.1, face_colors.2),
            (0, 2, 1) => (face_colors.0, face_colors.2, face_colors.1),
            (1, 0, 2) => (face_colors.1, face_colors.0, face_colors.2),
            (1, 2, 0) => (face_colors.1, face_colors.2, face_colors.0),
            (2, 0, 1) => (face_colors.2, face_colors.0, face_colors.1),
            (2, 1, 0) => (face_colors.2, face_colors.1, face_colors.0),
            _ => {
                assert!(false, "BUG: Somethig very off about face colors");
                (face_colors.0, face_colors.1, face_colors.2)
            }
        }
    }

    fn get_face_colors_edge<'a>(
        face_colors: (&'a FaceColor, &'a FaceColor),
        color_indices: (u32, u32),
    ) -> (&'a FaceColor, &'a FaceColor) {
        match color_indices {
            (0, 1) => (face_colors.0, face_colors.1),
            (1, 0) => (face_colors.1, face_colors.0),
            _ => {
                assert!(false, "BUG: Somethig very off about face colors");
                (face_colors.0, face_colors.1)
            }
        }
    }

    fn get_color_representation(color: &FaceColor) -> Vec<f32> {
        match color {
            FaceColor::Red => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            FaceColor::Blue => {
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            }
            FaceColor::Green => {
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            }
            FaceColor::White => {
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            }
            FaceColor::Yellow => {
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            }
            FaceColor::Orange => {
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
        }
    }

    fn get_cubelet_representation(
        slotted_cubelet: &SlottedCubelet,
        cubelets: &[Cubelet; 26],
    ) -> Option<tch::Tensor> {
        //BUG HERE: The representation is not checking the order of colors
        match slotted_cubelet {
            SlottedCubelet::Center(i) => {
                let cubelet = &cubelets[i.clone() as usize];
                match cubelet {
                    Cubelet::Center(i) => {
                        // tch::Tensor::empty(&[1,1],(tch::Kind::Float,tch::Device::Mps));
                        Some(tch::Tensor::from_slice(&Self::get_color_representation(i)))
                    }
                    _ => panic!("Expected corner cubelet"),
                }
            }
            SlottedCubelet::Corner(c, i) => {
                let cubelet = &cubelets[i.clone() as usize];
                let color_indices = c.get_raw_color_indices();
                match cubelet {
                    Cubelet::Corner(i, j, k) => {
                        let face_colors = Self::get_face_colors_corner((i, j, k), color_indices);
                        let mut color_reps = Self::get_color_representation(face_colors.0);
                        color_reps.append(&mut Self::get_color_representation(face_colors.1));
                        color_reps.append(&mut Self::get_color_representation(face_colors.2));
                        Some(tch::Tensor::from_slice(&color_reps))
                    }
                    _ => panic!("Expected corner cubelet"),
                }
            }
            SlottedCubelet::Edge(c, i) => {
                let cubelet = &cubelets[i.clone() as usize];
                let color_indices = c.get_raw_color_indices();
                match cubelet {
                    Cubelet::Edge(i, j) => {
                        let face_colors = Self::get_face_colors_edge((i, j), color_indices);
                        let mut color_reps = Self::get_color_representation(face_colors.0);
                        color_reps.append(&mut Self::get_color_representation(face_colors.1));
                        Some(tch::Tensor::from_slice(&color_reps))
                    }
                    _ => panic!("Expected corner cubelet"),
                }
            }
        }
    }

    fn get_face_strings() -> [String; 26] {
        [
            "right".to_string(),
            "left".to_string(),
            "up".to_string(),
            "down".to_string(),
            "back".to_string(),
            "front".to_string(),
            "left-up".to_string(),
            "left-down".to_string(),
            "left-back".to_string(),
            "left-front".to_string(),
            "right-up".to_string(),
            "right-down".to_string(),
            "right-back".to_string(),
            "right-front".to_string(),
            "up-back".to_string(),
            "up-front".to_string(),
            "down-back".to_string(),
            "down-front".to_string(),
            "up-left-front".to_string(),
            "up-right-front".to_string(),
            "up-left-back".to_string(),
            "up-right-back".to_string(),
            "down-left-front".to_string(),
            "down-right-front".to_string(),
            "down-left-back".to_string(),
            "down-right-back".to_string(),
        ]
    }

    pub fn index_cube_move(index: u32) -> Option<CubeMove> {
        match index {
            0 => Some(CubeMove::LPlus),
            1 => Some(CubeMove::LMinus),
            2 => Some(CubeMove::RPlus),
            3 => Some(CubeMove::RMinus),
            4 => Some(CubeMove::UPlus),
            5 => Some(CubeMove::UMinus),
            6 => Some(CubeMove::DPlus),
            7 => Some(CubeMove::DMinus),
            8 => Some(CubeMove::FPlus),
            9 => Some(CubeMove::FMinus),
            10 => Some(CubeMove::BPlus),
            11 => Some(CubeMove::BMinus),
            12 => Some(CubeMove::NoOp),
            _ => None,
        }
    }

    fn get_cube_move_index(mv: &CubeMove) -> u32 {
        match mv {
            CubeMove::LPlus => 0,
            CubeMove::LMinus => 1,
            CubeMove::RPlus => 2,
            CubeMove::RMinus => 3,
            CubeMove::UPlus => 4,
            CubeMove::UMinus => 5,
            CubeMove::DPlus => 6,
            CubeMove::DMinus => 7,
            CubeMove::FPlus => 8,
            CubeMove::FMinus => 9,
            CubeMove::BPlus => 10,
            CubeMove::BMinus => 11,
            CubeMove::NoOp => 12,
        }
    }

    fn get_cube_mov_rep(mv: &CubeMove) -> tch::Tensor {
        let mv_slice = match mv {
            CubeMove::LPlus => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::LMinus => {
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::RPlus => {
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::RMinus => {
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::UPlus => {
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::UMinus => {
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::DPlus => {
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::DMinus => {
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::BPlus => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            }
            CubeMove::BMinus => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            }
            CubeMove::FPlus => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            }
            CubeMove::FMinus => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            }
            CubeMove::NoOp => {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
        };
        Tensor::from_slice(mv_slice)
    }

    pub fn gen_input_representation(cube: &RubiksCube) -> tch::Tensor {
        let mut t = tch::Tensor::empty(&[0, 1], (tch::Kind::Float, tch::Device::Cpu));
        let cube_slot_map = cube.cube_slot_map.borrow();
        let face_strings = Self::get_face_strings();
        for face_string in face_strings {
            let s = cube_slot_map
                .get(&face_string)
                .expect("Expected cube slot to be available");
            let sc_t_o = Self::get_cubelet_representation(s, &cube.cubelets);
            if let Some(sc_t) = sc_t_o {
                // info!("Got cubelet {:?} {:?}",sc_t.dim(),sc_t);
                t = tch::Tensor::cat(&[t, sc_t.unsqueeze(1)], 0);
                // info!("Got cubelet {:?} {:?}",t.dim(),t);
            } else {
                info!(
                    "Warning; couldn't get cubelet representation. That wasn't supposed to happen."
                );
            }
        }
        // info!("Returning input tensor {:?}",t);
        t.to_device(tch::Device::Mps)
    }

    //The generated tuple is (a,s,p_a,r_s)
    //TODO: Return the logits as well for collection
    fn gen_trajectory(&self, cube_start: RubiksCube) -> Trajectory {
        let mut trajectory_moves = Vec::new();
        let mut trajectory_rewards = Vec::new();
        let mut trajectory_logits = Tensor::empty([0], (tch::Kind::Float, tch::Device::Mps));
        let mut trajectory_cube_states = Tensor::empty([0], (tch::Kind::Float, tch::Device::Mps));
        let mut current_cube = cube_start;
        for i in 0..self.trajectory_depth {
            let input = Self::gen_input_representation(&current_cube);
            let output = self.policy.forward(&input.transpose(0, 1));
            // info!("Generated these logits when generating trajectory: {:?} {:?}",
            // output.size(),output.print());
            let mv = Self::sample_action(&output);
            let cube_t = current_cube.apply_move(mv.clone());
            let r_t = Self::get_reward(&cube_t, &current_cube);
            // info!("Computed the mv & the o to be: {} for mv {:?}",r_t,mv);
            trajectory_moves.push(mv);
            trajectory_logits = tch::Tensor::concat(&[trajectory_logits, output], 0);
            trajectory_cube_states = 
                tch::Tensor::concat(
                    &[trajectory_cube_states,
                    Self::gen_input_representation(&cube_t)], 0);
            trajectory_rewards.push(r_t);
            current_cube = cube_t;
        }
        println!(
            "Generated trajectory: {:?} & rewards: {:?} & trajectory logits: {:?}",
            trajectory_moves,
            trajectory_rewards,
            trajectory_logits.size()
        );
        Trajectory::new(
                trajectory_cube_states, 
                trajectory_moves, 
                trajectory_logits, 
                trajectory_rewards)
    }

    pub fn gen_trajectories(&self, cube_start: Vec<RubiksCube>) -> Vec<Trajectory> {
        let mut trajs = Vec::new();
        for i in 0..self.num_trajectories {
            //TODO: Can we avoid this clone?
            let traj = self.gen_trajectory(
                cube_start
                    .get(i as usize)
                    .expect("Expected start cube at position")
                    .clone(),
            );
            // info!("Generated the trajectory:{:?}",traj.0);
            trajs.push(traj);
        }
        trajs
    }

    //Shape of the logits: [num_trajectory, trajectory_depth, num_moves]
    //Shape of the moves is [num_trajectory, trajectory_depth]
    //Output tensor is [num_trajectory, trajectory_depth, 1]
    fn log_probs_policy_su(logits: &Tensor, mv: &Tensor) -> Tensor {
        // logits.get(Self::get_cube_move_index(&mv)).log()
        // info!("Move tensor on {:?} {:?}",mv.device(),mv);
        // info!("Logits tensor on {:?} {:?}",logits.device(),logits);
        info!(
            "Size of logits & moves {:?} {:?} {:?}",
            logits.size(),
            mv.size(),
            mv.unsqueeze(2)
        );
        let mps_mv = mv.to_device(tch::Device::Mps);
        let log_logits_m = logits.gather(2, &mps_mv.unsqueeze(2), false).log();
        //.sum(tch::Kind::Float);
        info!("Size of sampled logits tensor {:?}", log_logits_m.size()); //,log_logits_m);
        log_logits_m
    }

    //log_probs shape:  [num_trajectory, trajectory_depth, 1]
    //Rewards shape: [num_trajectory, trajectory_depth]
    //Output: [1]
    fn expected_policy_reward_su(log_probs: Tensor, rewards: Tensor) -> Tensor {
        info!(
            "Tensors for calculating policy loss, log_probs: {:?} & rewards: {:?}",
            log_probs.size(),
            rewards.size()
        );
        info!("Unweighted rewards: {:?}", rewards.size());
        let mps_rewards = rewards.to_device(tch::Device::Mps);
        // rewards.print();
        //Summing the log probabilities for each trajectory:
        let weighted_rewards =
            (log_probs.squeeze_dim(2) * mps_rewards).sum_dim_intlist(1, false, tch::Kind::Float);
        info!("Weighted rewards: {:?}", weighted_rewards.size());
        // weighted_rewards.print();
        -weighted_rewards.mean_dim(0, true, tch::Kind::Float)
    }

    fn compute_mean_traj_entropy(trajs: &Vec<Trajectory>) -> f32 {
        let mut cl = Vec::new();
        for traj in trajs {
            cl.push(traj.compute_avg_entropy());
        }
        cl.iter().sum::<f32>() / (cl.len() as f32)
    }

    //Running the training simulation involves for an epoc
    //Involves collection of n trajectories with m moves
    pub fn train_an_epoch_vpg(&mut self, trajs: Vec<Trajectory>) {
        let rewards_t = Self::get_rewards_tensor(&trajs);
        let mv_t = Self::get_moves_tensor(&trajs);

        //Get input tensor  of shape [num_trajectories, traj_depth, input]
        let all_traj_logits_l: Vec<&Tensor> =
            trajs.iter().map(|trajectory| &trajectory.logits).collect();
        let all_traj_logits: Tensor = Tensor::stack(&all_traj_logits_l, 0);

        //Run the backward prop
        let expected_reward = Self::expected_policy_reward_su(
            Self::log_probs_policy_su(&all_traj_logits, &mv_t),
            rewards_t,
        );
        //TODO: Add term for mean entropy regularalization as well:
        info!(
            "Loss tensor: {:?} {:?}",
            expected_reward.size(),
            expected_reward.to_device(tch::Device::Cpu)
        );
        info!(
            "Mean epoch policy entropy for the trajectories is: {:?}",
            Self::compute_mean_traj_entropy(&trajs)
        );
        self.optim.zero_grad();
        expected_reward.backward();
        self.optim.step();
    }

    fn get_rewards_tensor(trajs: &Vec<Trajectory>) -> Tensor {
        //Collect rewards tensor
        let rewards: Vec<f32> = trajs
            .iter()
            .flat_map(|trajectory| trajectory.rewards.clone())
            .collect();
        let rewards_t = Tensor::f_from_slice(&rewards)
            .unwrap()
            .reshape(&[self.num_trajectories as i64, self.trajectory_depth as i64]);
        rewards_t
    }

    fn get_moves_tensor(trajs: &Vec<Trajectory>) -> Tensor {
        //Get move tensor:
        let move_indices_l: Vec<Vec<i64>> = trajs
            .iter()
            .map(|trajectory| {
                trajectory
                    .moves
                    .iter()
                    .map(|f| Self::get_cube_move_index(f) as i64)
                    .collect::<Vec<i64>>()
            })
            .collect();
        let move_indices: Vec<i64> = move_indices_l.iter().flat_map(|e| e.clone()).collect();
        let mv_t = Tensor::f_from_slice(&move_indices)
            .unwrap()
            .reshape(&[self.num_trajectories as i64, self.trajectory_depth as i64]);
        mv_t
    }

    fn get_advantages_tensor(advnet: nn::Sequential, trajs: &Vec<Trajectory>) -> Tensor {
        trajs.iter().flat_map(|traj| {
            //Input tensor for advnet [rep_size + move_rep,1]
            let advantages = advnet.forward(xs);
        });
    }

    fn get_added_advantage_tensor(on_pol_adv: Tensor, trajs: &Vec<Trajectory>) -> Tensor {
        //Collect the prob-logits for the target policy moves of size [num_trajectories, trajectory, 1]
        //Collect the prob-logits for the trained policy moves of size 
        //Clip on the basis of added advantage

    }

    pub fn train_an_epoch_ppo(&mut self, trajs: Vec<Trajectory>) {
        //Get the advantage tensor for the trajectory:

        //Get the added advantage tensor for the trajectory:
        // GD on the loss for training policy:
        // GD on the mean square loss b/w the advantage & the rtg tensor to train
    }
}
