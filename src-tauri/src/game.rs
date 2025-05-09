use serde_big_array::BigArray;
use std::collections::HashSet;
use lazy_static::lazy_static;

pub type Card = usize;

#[derive(PartialEq, Eq, Debug, serde::Deserialize)]

pub enum Phase {
    //Init,
    Discard,
    DiscardPick,
    //Draw,
    DrawPick,
    KoiKoi,
    //RoundOver,
}

const DEFAULT_ROUND_TOTAL: u32 = 8;
const DEFAULT_INIT_POINT: u32 = 30;

const CRANE: [Card; 1] = [0];
const CURTAIN: [Card; 1] = [8];
const MOON: [Card; 1] = [28];
const RAIN_MAN:  [Card; 1] = [40];
const PHOENIX: [Card; 1] = [44];
const SAKE: [Card; 1] = [32];

const LIGHT: [Card; 5] = [0, 8, 28, 40, 44];
const SEED:  [Card; 9] = [4, 12, 16, 20, 24, 29, 32, 36, 41];
const RIBBON: [Card; 10] = [1, 5, 9, 13, 17, 21, 25, 33, 37, 42];
const DROSS: [Card; 25] = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31,
34, 35, 38, 39, 43, 45, 46, 47, 32];
        
const BOAR_DEER_BUTTERFLY: [Card; 3] = [20,24,36];
const FLOWER_SAKE: [Card; 2] = [8, 32];
const MOON_SAKE: [Card; 2] = [28, 32];
const RED_RIBBON: [Card; 3] = [1, 5, 9];
const BLUE_RIBBON: [Card; 3] = [21, 33, 37];
const RED_BLUE_RIBBON: [Card; 6] = [1, 5, 9, 21, 33, 37];

lazy_static! {
    pub static ref CARD_LIST: Vec<Vec<Card>> = vec![
        CRANE.to_vec(),
        CURTAIN.to_vec(),
        MOON.to_vec(),
        RAIN_MAN.to_vec(),
        PHOENIX.to_vec(),
        SAKE.to_vec(),
        BOAR_DEER_BUTTERFLY.to_vec(),
        SEED.to_vec(),
        RED_RIBBON.to_vec(),
        BLUE_RIBBON.to_vec(),
        RED_BLUE_RIBBON.to_vec(),
        RIBBON.to_vec(),
        DROSS.to_vec(),
    ];
}


#[derive(serde::Deserialize)]
pub struct F32_48 {
    #[serde(with = "BigArray")]
    pub arr: [f32; 48]
}

type CardLog = [[F32_48; 8]; 16];

#[derive(serde::Deserialize)]
pub struct GameState {
    pub round: usize,
    pub points: [i8; 2],

    pub hand: [Vec<Card>; 2],
    pub pile: [Vec<Card>; 2],
    pub field: Vec<Card>,
    pub stock: Vec<Card>,
    
    pub init_board: Vec<Card>,

    pub show: Vec<Card>,
    pub collected: Vec<Card>,
    
    pub turn_16: usize,
    pub dealer: usize,
    pub koikoi: [[i32; 8]; 2],
    winner: Option<usize>,
    exhausted: bool,
    turn_point: i32,
    
    pub phase: Phase,
    wait_action: bool,

    #[serde(with = "BigArray")]
    pub card_log: CardLog,
}

impl GameState {
    /*
    fn new(dealer: Option<usize>) -> Self {
        let hand = [vec!(), vec!()];
        let pile = [vec!(), vec!()];
        let field_slot = vec!();
        let stock = vec!();
        
        let show = vec!();
        let collect = vec!();
        
        let turn_16 = 1;
        let dealer = match dealer {
            Some(d) => d,
            None => 1, // todo
        };
        
        let koikoi = [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]];
        let winner = None;
        let exhausted = false;
        
        let turn_point = 0;
        
        let state = State::Init;
        let wait_action = false;

        Self {
            hand,
            pile,
            field_slot,
            stock,
            show,
            collect,
            turn_16,
            dealer,
            koikoi,
            winner,
            exhausted,
            turn_point,
            state,
            wait_action,
        } 
    }
    */

    fn new_round(&mut self) {

    }

    pub fn turn_player(&self) -> usize {
        if (self.turn_16+self.dealer)%2==1 {0} else {1}
    }

    fn turn_8(&self) -> usize {
        (self.turn_16+1)/2
    }

    /*
    pub fn field(&self) -> Vec<Card> {
        let mut res: Vec<_> = self.field_slot
            .iter()
            .filter(|(x, y)| *x != 0 || *y != 0)
            .copied()
            .collect();
        res.sort_unstable();
        res
    }
    */

    pub fn unseen_cards(&self, player: usize) -> Vec<Card> {
        let mut unseen = self.stock.clone();
        if player == 0 {
            unseen.extend(&self.hand[1]);
        } else {
            unseen.extend(&self.hand[0]);
        }
        unseen
    }

    pub fn pairing_cards(&self) -> Vec<Card> {
        self.field.iter().filter(|&&c| c/4 == self.show[0] / 4).copied().collect()
    }

    /*
    fn field_collect(&self) -> Vec<Card> {
        self.collected
            .iter()
            .filter(|&&c| c != self.show[0])
            .copied()
            .collect()
    }

    fn round_points(&self, player: usize) -> Option<i32> {
        if self.winner.is_none() {
            None
        } else if self.exhausted {
            if (self.dealer==0) == (player == 0) { Some(1) } else { Some(-1) }
        } else if self.winner == Some(0) {
            if player == 0 {Some(self.yaku_points(0))} else {Some(-self.yaku_points(0))}
        } else {
            if player == 1 {Some(self.yaku_points(1))} else {Some(-self.yaku_points(1))}    
        }
    }
    */

    pub fn koikoi_num(&self, player: usize) -> i32 {
        self.koikoi[player].iter().sum()
    }


    pub fn yaku_points(&self, player: usize) -> i32 {
        let mut point = 0; // todo
        /* 
             self.yaku(player)
            .filter()
            .map

        sum([yaku[2] for yaku in self.yaku(player) if yaku[1]!='Koi-Koi'])
        */
        let koikoi_num = self.koikoi_num(player);
        if koikoi_num <= 3 {
            point += koikoi_num
        }
        else {
            point *= koikoi_num - 2
    
        }
        point
    }

    /* 
    fn _deal_card(&mut self) {
        loop {
            let mut cards: Vec<Card> = vec!();
            for i in 1..13 {
                for j in 1..5 {
                    cards.push((i, j));
                }
            }
            //random.shuffle(card);   todo
            let mut hand1 = cards[0..8].to_vec();
            hand1.sort_unstable();
            let mut hand2 = cards[8..16].to_vec();
            hand2.sort_unstable();
            self.hand = [hand1, hand2];
            let mut field_slot = cards[16..24].to_vec();
            field_slot.sort_unstable();
            for i in 0..10 {
                field_slot.push((0, 0));
            }
            self.field_slot = field_slot;
            self.stock = cards[24..].to_vec();     
            let mut flag = true;
            for suit in 1..13 {
                /* 
                if 4 in [[card[0] for card in self.hand[1]].count(suit),
                         [card[0] for card in self.hand[2]].count(suit),
                         [card[0] for card in self.field].count(suit)]:
                    flag = false; 
                */
                flag = false;
            }
            if flag {
                break
            }
        }   
        self.state = State::Discard;
        self.wait_action = true;
        return
    }
    */

    /*
    fn _collect_card(&mut self, card: Card) {
        let pairing_card = self.pairing_cards();
        let n = pairing_card.len();
        if pairing_card.is_empty() {
            self.collect = Vec::new();
            if let Some(index) = self.field_slot.iter().position(|&slot| slot == (0, 0)) {
                self.field_slot[index] = self.show[0];
            }
        }
        else if n == 1 || n == 3 {
            self.collect = self.show.clone();
            self.collect.extend(&pairing_card);
            for paired_card in &pairing_card {
                if let Some(index) = self.field_slot.iter().position(|&slot| slot == *paired_card) {
                    self.field_slot[index] = (0, 0);
                }
            };
            self.pile[self.turn_player()].extend(&self.collect);
        }
        else {
            self.collect = self.show.clone();
            self.collect.push(card);
            if let Some(index) = self.field_slot.iter().position(|&slot| slot == card) {
                self.field_slot[index] = (0, 0);
            }
            self.pile[self.turn_player()].extend(&self.collect);
        }
    }

    fn discard(&mut self, card: Card) {
        let turn_player = self.turn_player();
        // Vérification des préconditions
        // assert_eq!(self.state, State::Discard);
        assert!(self.hand[turn_player].contains(&card));

        self.turn_point = self.yaku_points(turn_player);
        if let Some(ind) = self.hand[turn_player].iter().position(|&c| c == card) {
            self.show = vec![self.hand[turn_player].remove(ind)];
        }

        self.state = State::DiscardPick;
        self.wait_action = self.pairing_cards().len() == 2;

        // Retourner l'état ou appeler __call__
        /*
        if self.silence {
            self.state.clone()
        } else {
            self.call()
        }
        */
    }

    fn discard_pick(&mut self, card: Option<Card>) {
        /*assert_eq!(self.state, State.DiscardPick);
        
        if self.wait_action {
            assert!(card.is_some() && self.pairing_card.contains(&card.unwrap()));
        } else {
            assert!(card.is_none());
        }
        */

        if let Some(c) = card {
            self._collect_card(c);
        }

        self.state = State::Draw;
        self.wait_action = false;
    }    

    fn draw(&mut self) {
        //assert_eq!(self.state, "draw");

        if let Some(c) = self.stock.pop() {
            self.show = vec![c];
        }

        self.state = State::DrawPick;
        self.wait_action = self.pairing_cards().len() == 2;
    }

    fn draw_pick(&mut self, card: Card) {
        //assert self.state == 'draw-pick'
        //assert (card in self.pairing_card) if self.wait_action else (card == None)
        
        self._collect_card(card);

        self.state = State::KoiKoi;
        self.wait_action = (self.yaku_points(self.turn_player()) > self.turn_point) && (self.turn_8() < 8);   
    }

    fn claim_koikoi(&mut self, mut is_koikoi: Option<bool>) {
        let turn_player = self.turn_player();
        let turn_8 = self.turn_8();
        // Action
        if self.yaku_points(turn_player) > self.turn_point && turn_8 == 8 {
            is_koikoi = Some(false);
        }
        self.koikoi[turn_player][turn_8 as usize - 1] = if is_koikoi.unwrap_or(false) { 1 } else { 0 };

        if is_koikoi == Some(false) {
            self.state = State::RoundOver;
            self.wait_action = false;
            self.winner = Some(turn_player);
        } else if self.turn_16 == 16 {
            self.state = State::RoundOver;
            self.wait_action = false;
            self.exhausted = true;
            self.winner = Some(self.dealer);
        } else {
            self.turn_16 += 1;
            self.state = State::Discard;
            self.wait_action = true;
        }
    }
    */

    fn yaku(&self, player: usize) -> Vec<(i32, &'static str, i32)> {
        let mut yaku = Vec::new();
        let pile: HashSet<Card> = self.pile[player].iter().cloned().collect();
        let koikoi_num = self.koikoi_num(player);

        let num_light = LIGHT.iter().filter(|c| pile.contains(c)).count();
        
        if num_light == 5 {
            yaku.push((1, "Five Lights", 10));
        } else if num_light == 4 && !pile.contains(&40) {
            yaku.push((2, "Four Lights", 8));
        } else if num_light == 4 {
            yaku.push((3, "Rainy Four Lights", 7));
        } else if num_light == 3 && !pile.contains(&40) {
            yaku.push((4, "Three Lights", 5));
        }

        let num_seed = SEED.iter().filter(|c| pile.contains(c)).count();
        if BOAR_DEER_BUTTERFLY.iter().all(|c| pile.contains(c)) {
            yaku.push((5, "Boar-Deer-Butterfly", 5));
        }
        if FLOWER_SAKE.iter().all(|c| pile.contains(c)) {
            if koikoi_num == 0 {
                yaku.push((6, "Flower Viewing Sake", 1));
            } else {
                yaku.push((7, "Flower Viewing Sake", 3));
            }
        }
        if MOON_SAKE.iter().all(|c| pile.contains(c)) {
            if koikoi_num == 0 {
                yaku.push((8, "Moon Viewing Sake", 1));
            } else {
                yaku.push((9, "Moon Viewing Sake", 3));
            }
        }
        if num_seed >= 5 {
            yaku.push((10, "Tane", (num_seed - 4) as i32));
        }

        let num_ribbon = RIBBON.iter().filter(|c| pile.contains(c)).count();
        
        // TODO a verifier les if / else
        if RED_RIBBON.iter().all(|c| pile.contains(c)) {
            if BLUE_RIBBON.iter().all(|c| pile.contains(c)) {
                yaku.push((11, "Red & Blue Ribbons", 10));
            } else  {
                yaku.push((12, "Red Ribbons", 5));
            }
        } else if BLUE_RIBBON.iter().all(|c| pile.contains(c)) {
            yaku.push((13, "Blue Ribbons", 5));
        }
        if num_ribbon >= 5 {
            yaku.push((14, "Tan", (num_ribbon - 4) as i32));
        }

        let num_dross = DROSS.iter().filter(|c| pile.contains(c)).count();
        if num_dross >= 10 {
            yaku.push((15, "Kasu", (num_dross - 9) as i32));
        }

        if koikoi_num > 0 {
            yaku.push((16, "Koi-Koi", koikoi_num as i32));
        }

        yaku
    }

    /*
    fn call(&self, view: Option<usize>) {
        let view = view.unwrap_or(self.turn_player());
        let op_view = 1 - view;
        let pile: HashSet<Card> = self.pile[view].iter().cloned().collect();
        let op_pile: HashSet<Card> = self.pile[op_view].iter().cloned().collect();

        println!("Turn: {},  State: {:?}", self.turn_8(), self.state);
        println!("-----------------------------------------------");
        println!("Opponent's Yaku:");
        let op_yaku: Vec<(i32, &str, i32)> = self.yaku(op_view);
        for yaku in &op_yaku {
            println!("[{}, {}]", yaku.1, yaku.2);
        }
        println!("Total Point: {}", self.yaku_points(op_view));
        println!("-----------------------------------------------");
        println!("Opponent's Pile:");
        println!("Light: {:?}", LIGHT.iter().filter(|c| op_pile.contains(c)).count());
        println!("Seed: {:?}", SEED.iter().filter(|c| op_pile.contains(c)).count());
        println!("Ribbon: {:?}", RIBBON.iter().filter(|c| op_pile.contains(c)).count());
        println!("Dross: {:?}", DROSS.iter().filter(|c| op_pile.contains(c)).count());
        println!("-----------------------------------------------");
        println!("Opponent's Hand:");
        for _card in &self.hand[op_view] {
            println!("[0, 0]");
        }
        println!("-----------------------------------------------");
        println!("Field:");
        println!("{:?}", self.field());
        println!("-----------------------------------------------");
        println!("Your Hand:");
        println!("{:?}", self.hand[view]);
        println!("-----------------------------------------------");
        println!("Your Pile:");
        println!("Light: {:?}", LIGHT.iter().filter(|c| pile.contains(c)).count());
        println!("Seed: {:?}", SEED.iter().filter(|c| pile.contains(c)).count());
        println!("Ribbon: {:?}", RIBBON.iter().filter(|c| pile.contains(c)).count());
        println!("Dross: {:?}", DROSS.iter().filter(|c| pile.contains(c)).count());
        println!("-----------------------------------------------");
        println!("Your Yaku:");
        let your_yaku: Vec<(i32, &str, i32)> = self.yaku(view);
        for yaku in &your_yaku {
            println!("[{}, {}]", yaku.1, yaku.2);
        }
        println!("Total Point: {}", self.yaku_points(view));
        println!("-----------------------------------------------");

        if view != self.turn_player() {
            println!("Opponent's turn, waiting action...");
            return;
        }

        match self.state {
            State::Discard => {
                println!("Use discard(card) to discard from hand.");
            }
            State::DiscardPick => {
                println!("Discard: {:?}", self.show[0]);
                println!("Pairing: {:?}", self.pairing_cards());
                if self.wait_action {
                    println!("Use discard_pick(card) to pick a pairing field card.");
                } else {
                    println!("Use discard_pick() to continue.");
                }
            }
            State::Draw => {
                println!("Use draw() to draw from stock.");
            }
            State::DrawPick => {
                println!("Draw: {:?}", self.show[0]);
                println!("Pairing: {:?}", self.pairing_cards());
                if self.wait_action {
                    println!("Use draw_pick(card) to pick a pairing field card.");
                } else {
                    println!("Use draw_pick() to continue.");
                }
            }
            State::KoiKoi => {
                if self.wait_action {
                    println!("Use claim_koikoi(bool) to koikoi or stop.");
                } else {
                    println!("Use claim_koikoi() to continue.");
                }
            }
            State::RoundOver => {
                println!("Round Over");
                println!("Round Point: You {:?}, Opponent {:?}", self.round_points(view), self.round_points(op_view));
            }
            _ => {}
        }
    }
    */
}


/*
#[derive(serde::Deserialize)]
pub struct GameState {
    pub round_total: usize,
    pub init_points: [usize; 2],
    pub init_dealer: Option<usize>,
    pub round_state: RoundState,
    pub round: usize,
    pub points: [usize; 2], 
    pub game_over: bool,
    pub winner: Option<usize>
}
*/
