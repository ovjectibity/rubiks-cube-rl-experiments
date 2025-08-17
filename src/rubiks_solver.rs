use rand::Rng;
use tch::{nn,Tensor,nn::Module,nn::Adam,nn::OptimizerConfig,no_grad};
use crate::rubiks::{CornerSlot, CubeMove, Cubelet, FaceColor, RubiksCube, SlottedCubelet};

pub struct Trajectory {
    moves: Vec<CubeMove>,
    //Shape of logits: [trajectory_depth,num_possible_moves]
    logits: Tensor,
    rewards: Vec<f32>
}

impl Trajectory {
    fn new(moves: Vec<CubeMove>,logits: Tensor,rewards: Vec<f32>) -> Self {
        let mut traj = Trajectory { 
            moves: moves, 
            logits: logits, 
            rewards: rewards
        };
        traj.compute_rtg();
        traj
    }

    fn compute_rtg(&mut self) {
        let mut acc = 0.0;
        let mut rtgs: Vec<f32> = self.rewards.iter().rev().map(|reward| {
            acc += reward;
            acc
        }).collect();
        rtgs.reverse();
        self.rewards = rtgs;
    }
}

pub struct RubiksSolver {
    policy: nn::Sequential,
    trajectory_depth: u32,
    num_trajectories: u32,
    num_layers: u32,
    hidden_layer_dimension: u32,
    pub num_epochs: u32,
    store: nn::VarStore,
    optim: nn::Optimizer
}

//Uses the policy gradient algorithm
impl RubiksSolver {
    pub fn new(num_layers: u32,hidden_layer_dimension: u32,
            num_epochs: u32,num_trajectories: u32,trajectory_depth: u32,
            learning_rate: f64) -> Self {
        let pols = Self::init_policy(hidden_layer_dimension as i64,num_layers as i64);
        let optim = nn::Adam::default().build(&pols.1,learning_rate).unwrap();
        //Initialise the tensors uniformly in the policy network: 
        no_grad(|| {
            for (name, mut tensor) in pols.1.variables() {
                if name.ends_with("weight") {
                    tensor.uniform_(-0.05, 0.05);     // in-place
                } else if name.ends_with("bias") {
                    tensor.zero_();
                }
            }
        });

        RubiksSolver {
            policy: pols.0,
            num_layers: num_layers,
            hidden_layer_dimension: hidden_layer_dimension,
            num_epochs: num_epochs,
            num_trajectories: num_trajectories,
            trajectory_depth: trajectory_depth,
            store: pols.1,
            optim: optim,
        }
    }

    fn init_policy(hidden_layer_dimension: i64,num_layers: i64) -> (nn::Sequential,nn::VarStore) {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vs_p = vs.root();
        let mut y = nn::seq().add(nn::linear(vs_p.clone(), 54, hidden_layer_dimension, Default::default())).
            add_fn(Tensor::relu);
        for i in 0..num_layers-2 {
            y = y.add(nn::linear(vs_p.clone(),hidden_layer_dimension,hidden_layer_dimension,Default::default())).
            add_fn(Tensor::relu);
        }
        y = y.add(nn::linear(vs_p.clone(),hidden_layer_dimension,13,Default::default())).
            add_fn(Tensor::relu).
            add_fn(|xs| {
                xs.softmax(-1, tch::Kind::Float)
            });
        (y,vs)
    }

    pub fn generate_move_logits(&self,cube: &RubiksCube) -> Tensor {
        let input = Self::gen_input_representation(cube);
        // println!("Generated input tensor: {:?} {:?}",input.dim(),input);
        // println!("Policy tensor: {:?} {:?}",self.policy,&self.policy);
        let output = self.policy.forward(&input.transpose(0,1));
        output
    }

    pub fn generate_move(&self,cube: &RubiksCube) -> CubeMove {
        let input = Self::gen_input_representation(cube);
        // println!("Generated input tensor: {:?} {:?}",input.dim(),input);
        // println!("Policy tensor: {:?} {:?}",self.policy,&self.policy);
        let output = self.policy.forward(&input.transpose(0,1));
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
            let tar_cubelet = 
                scm.get(&face_string).expect("Expected slotted cubelet");
            let sol_cubelet = 
                solved_scm.get(&face_string).expect("Expected slotted cubelet");
            match (tar_cubelet,sol_cubelet) {
                (SlottedCubelet::Corner(c1,i1),SlottedCubelet::Corner(c2,i2)) => {
                    // println!("Cubelet & color indices for the compared corner cubelets: {:?} {:?} {:?} {:?}",
                    //         i1,i2,c1.get_raw_color_indices(),c2.get_raw_color_indices());
                    if i1 == i2 && c1.get_raw_color_indices() == c2.get_raw_color_indices() {
                        score += 1.0;
                    } 
                    // else if i1 == i2 {
                    //     score += 0.5;
                    // }
                },
                (SlottedCubelet::Edge(c1,i1),SlottedCubelet::Edge(c2,i2)) => {
                    // println!("Cubelet & color indices for the compared edge cubelets: {:?} {:?} {:?} {:?}",
                    //         i1,i2,c1.get_raw_color_indices(),c2.get_raw_color_indices());
                    if i1 == i2 && c1.get_raw_color_indices() == c2.get_raw_color_indices()  {
                        score += 1.0;
                    } 
                    // else if i1 == i2 {
                    //     score += 0.5;
                    // }
                },
                (SlottedCubelet::Center(i1),SlottedCubelet::Center(i2)) => {
                    //do nothing
                },
                _ => panic!("Unexpected cubelet combo")
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

    fn sample_action(probs: &Tensor) -> CubeMove {
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

    fn get_color_representation(color: &FaceColor) -> f32 {
        match color {
            FaceColor::Red => {
                1.0
            },
            FaceColor::Blue => {
                2.0
            },
            FaceColor::Green => {
                3.0
            }, 
            FaceColor::White => {
                4.0
            },
            FaceColor::Yellow => {
                5.0
            },
            FaceColor::Orange => {
                6.0
            }
        }
    }

    fn get_cubelet_representation(
        slotted_cubelet: &SlottedCubelet,
        cubelets: &[Cubelet;26]) -> Option<tch::Tensor> {
        match slotted_cubelet {
            SlottedCubelet::Center(i) => {
                let cubelet = &cubelets[i.clone() as usize];
                match cubelet {
                    Cubelet::Center(i) => {
                        // tch::Tensor::empty(&[1,1],(tch::Kind::Float,tch::Device::Cpu));
                        Some(tch::Tensor::from_slice(&[Self::get_color_representation(i)]))
                    },
                    _ => None
                }
            },
            SlottedCubelet::Corner(c,i) => {
                let cubelet = &cubelets[i.clone() as usize];
                match cubelet {
                    Cubelet::Corner(i,j,k) => {
                        Some(tch::Tensor::from_slice(&[Self::get_color_representation(i),
                                                    Self::get_color_representation(j),
                                                    Self::get_color_representation(k)]))
                    },
                    _ => None
                }
            },
            SlottedCubelet::Edge(c,i) => {
                let cubelet = &cubelets[i.clone() as usize];
                match cubelet {
                    Cubelet::Edge(i,j) => {
                        Some(tch::Tensor::from_slice(&[Self::get_color_representation(i),
                                                    Self::get_color_representation(j)]))
                    },
                    _ => None
                }
            }
        }
    }

    fn get_face_strings() -> [String;26] {
        ["right".to_string(),
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
        "down-right-back".to_string()]
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
            _ => None
        }
    }

    fn get_cube_move_index(mv: &CubeMove) -> u32 {
        match mv {
            CubeMove::LPlus => {
                0
            },
            CubeMove::LMinus => {
                1
            },
            CubeMove::RPlus => {
                2
            },
            CubeMove::RMinus => {
                3
            },
            CubeMove::UPlus => {
                4
            },
            CubeMove::UMinus => {
                5
            },
            CubeMove::DPlus => {
                6
            },
            CubeMove::DMinus => {
                7
            },
            CubeMove::FPlus => {
                8
            },
            CubeMove::FMinus => {
                9
            },
            CubeMove::BPlus => {
                10
            },
            CubeMove::BMinus => {
                11
            },
            CubeMove::NoOp => {
                12
            }
        }
    }

    fn gen_input_representation(cube: &RubiksCube) -> tch::Tensor {
        let mut t = tch::Tensor::empty(&[0,1],(tch::Kind::Float,tch::Device::Cpu));
        let cube_slot_map = cube.cube_slot_map.borrow();
        let face_strings = Self::get_face_strings();
        for face_string in face_strings {
            let s = cube_slot_map.get(&face_string).expect("Expected cube slot to be available");
            let sc_t_o = Self::get_cubelet_representation(s, &cube.cubelets);
            if let Some(sc_t) = sc_t_o {
                // println!("Got cubelet {:?} {:?}",sc_t.dim(),sc_t);
                t = tch::Tensor::cat(&[t,sc_t.unsqueeze(1)],0);
                // println!("Got cubelet {:?} {:?}",t.dim(),t);
            } else {
                println!("Warning; couldn't get cubelet representation. That wasn't supposed to happen.");
            }
        }
        // println!("Returning input tensor {:?}",t);
        t
    }

    //The generated tuple is (a,s,p_a,r_s)
    //TODO: Return the logits as well for collection
    fn gen_trajectory(&self,cube_start: RubiksCube) -> Trajectory {
        let mut trajectory_moves = Vec::new();
        let mut trajectory_rewards = Vec::new();
        let mut trajectory_logits = Tensor::empty([0], (tch::Kind::Float,tch::Device::Cpu));
        let mut current_cube = cube_start;
        for i in 0..self.trajectory_depth {
            let input = Self::gen_input_representation(&current_cube);
            let output = self.policy.forward(&input.transpose(0, 1));
            // println!("Generated these logits when generating trajectory: {:?}",output.size());
            // output.print();
            let mv = Self::sample_action(&output);
            let cube_t = current_cube.apply_move(mv.clone());
            let r_t = Self::get_reward(&cube_t,&current_cube);
            // println!("Computed the mv & the o to be: {} for mv {:?}",r_t,mv);
            trajectory_moves.push(mv);
            trajectory_logits = tch::Tensor::concat(&[trajectory_logits,output], 0);
            trajectory_rewards.push(r_t);
            current_cube = cube_t;
        }
        println!("Generated trajectory: {:?} & rewards: {:?} & trajectory logits: {:?}",
            trajectory_moves,
            trajectory_rewards,
            trajectory_logits.size());
        Trajectory::new(trajectory_moves,trajectory_logits,trajectory_rewards)
    }

    //Shape of the logits: [num_trajectory, trajectory_depth, num_moves]
    //Shape of the moves is [num_trajectory, trajectory_depth]
    //Output tensor is [num_trajectory, trajectory_depth, 1]
    fn log_probs_policy_su(logits: &Tensor,mv: &Tensor) -> Tensor {
        // logits.get(Self::get_cube_move_index(&mv)).log()
        println!("Size of logits & moves {:?} {:?} {:?}",logits.size(),mv.size(),mv.unsqueeze(2));
        // mv.print();
        // logits.print();
        let log_logits_m = logits.gather(2,&mv.unsqueeze(2),false).log();
        //.sum(tch::Kind::Float);
        println!("Size of sampled logits tensor {:?}",log_logits_m.size()); 
        // log_logits_m.print();
        log_logits_m
    }

    //log_probs shape:  [num_trajectory, trajectory_depth, 1]
    //Rewards shape: [num_trajectory, trajectory_depth]
    //Output: [1]
    fn expected_policy_reward_su(log_probs: Tensor, rewards: Tensor) -> Tensor {
        println!("Tensors for calculating policy loss, log_probs: {:?} & rewards: {:?}",
                log_probs.size(),rewards.size());
        println!("Unweighted rewards: {:?}",rewards.size());
        // rewards.print();
        //Summing the log probabilities for each trajectory: 
        let weighted_rewards = 
            (log_probs.squeeze_dim(2) * rewards).sum_dim_intlist(1,false,tch::Kind::Float);
        println!("Weighted rewards: {:?}",weighted_rewards.size());
        // weighted_rewards.print();
        - weighted_rewards.mean_dim(0,true,tch::Kind::Float)
    }

    pub fn gen_trajectories(&self,cube_start: Vec<RubiksCube>) -> 
        Vec<Trajectory> {
        let mut trajs = Vec::new();
        for i in 0..self.num_trajectories {
            //TODO: Can we avoid this clone?
            let traj = self.gen_trajectory(cube_start.get(i as usize).
                expect("Expected start cube at position").clone());
            // println!("Generated the trajectory:{:?}",traj.0);
            trajs.push(traj);
        }
        trajs
    }

    //Running the training simulation involves for an epoc
    //Involves collection of n trajectories with m moves
    //Render those moves, with each new traj refresh all data 
    //Showcase that in the window 
    pub fn train_an_epoch(&mut self,trajs: Vec<Trajectory>) {
        //Collect rewards tensor
        let rewards: Vec<f32> = trajs.iter().flat_map(|trajectory| {
            trajectory.rewards.clone()
        }).collect();
        let rewards_t = 
            Tensor::f_from_slice(&rewards).
            unwrap().
            reshape(&[self.num_trajectories as i64,self.trajectory_depth as i64]);

        //Get move tensor:  
        let move_indices_l: Vec<Vec<i64>> = trajs.iter().map(|trajectory| {
            trajectory.moves.iter().map(|f| {
                Self::get_cube_move_index(f) as i64
            }).collect::<Vec<i64>>()
        }).collect();
        let move_indices: Vec<i64> = move_indices_l.iter().flat_map(|e| {
            e.clone()
        }).collect();
        let mv_t = 
            Tensor::f_from_slice(&move_indices).
            unwrap().
            reshape(&[self.num_trajectories as i64,self.trajectory_depth as i64]);
        
        //Get input tensor  of shape [num_trajectories, traj_depth, input]
        let all_traj_logits_l: Vec<&Tensor> = trajs.iter().map(|trajectory| {
            &trajectory.logits
        }).collect();
        let all_traj_logits: Tensor = Tensor::stack(&all_traj_logits_l, 0);

        //Run the backward prop 
        //BUG HERE: the starting point is not being used, instead starting point is the next step
        let expected_reward = Self::expected_policy_reward_su(
            Self::log_probs_policy_su(&all_traj_logits, &mv_t), rewards_t);
        println!("Loss tensor: {:?}",expected_reward.size());
        expected_reward.print();
        self.optim.zero_grad();
        expected_reward.backward();
        self.optim.step();
    }
}