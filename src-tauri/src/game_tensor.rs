use std::{array::from_fn, collections::HashSet};

use ndarray::prelude::*;
use burn::prelude::*;
use crate::game::{Card, CARD_LIST, Phase, GameState};
// use ndarray_npy::read_npy;

fn card_to_multi_hot(card_list: &[Card]) -> [f32; 48] {
    let mut card_multi_hot = [0f32; 48];
    for &c in card_list {
        card_multi_hot[c] = 1f32;
    }
    card_multi_hot
}

fn reserve_array() -> Array2<f32> {
    Array::zeros((17, 48))
}

fn inter_len(slice: &[Card], set: &HashSet<Card>) -> usize {
    slice.iter().filter(|card| set.contains(card)).count()
}

fn feature_tuple<const N: usize>(x: f32, power: [f32; N], weight: [f32; N]) -> [f32; N] { 
    from_fn(|i| x.powf(power[i]) * x.signum() * weight[i])
    // todo a verifier
    //ndarray.abs(float(x)) ** np.array(power) * np.sign(x) * np.array(weight)
}

fn feature_one_hot(pos: usize, feature_length: usize) -> Vec<f32> {
    let mut x = vec![0.; feature_length];
    x[pos] = 1.;
    x
}

fn game_status_array(state: &GameState) -> Array2<f32> {
    let turn_player = state.turn_player();
    let idle_player = 1 - turn_player;
        
    let point_diff = (state.points[turn_player] - state.points[idle_player]) as f32;
        
    let game_points = feature_tuple(point_diff/2., [0.5,1.,1.5], [1.,0.5,0.1]);
    let my_yaku_points = feature_tuple(
            state.yaku_points(turn_player) as f32, [0.5,1.,1.5], [1.,0.5,0.1]);

    let op_yaku_points = feature_tuple(
            state.yaku_points(idle_player) as f32, [0.5,1.,1.5], [1.,0.5,0.1]);
        
    let round =  feature_one_hot(state.round-1, 8);
    let turn = feature_one_hot(state.turn_16-1, 16);
    let dealer = feature_one_hot(state.dealer, 2);
        
    let my_koikoi_num = feature_tuple(
            state.koikoi_num(turn_player) as f32, [1.,2.], [1.,1.]);
    let op_koikoi_num = feature_tuple(
            state.koikoi_num(idle_player) as f32, [1.,2.], [1.,1.]);
        
    let my_koikoi = state.koikoi[turn_player].map(|x| x as f32);
    let op_koikoi = state.koikoi[idle_player].map(|x| x as f32);
    
    let f_array = [
        game_points.as_slice(),
        my_yaku_points.as_slice(),
        op_yaku_points.as_slice(),
        round.as_slice(),
        turn.as_slice(),
        dealer.as_slice(),
        my_koikoi_num.as_slice(),
        op_koikoi_num.as_slice(),
        my_koikoi.as_slice(),
        op_koikoi.as_slice()
    ].concat();
    let f_array = Array1::from(f_array);
    f_array
        .broadcast((48, f_array.len()))
        .unwrap()
        .t()
        .to_owned()
}

fn yaku_status_array(state: &GameState) -> Array2<f32> {
    let turn_player = state.turn_player();
    let idle_player = 1 - turn_player;

    let my_hand_cards: HashSet<_> = state.hand[turn_player].iter().copied().collect();
    let board_cards: HashSet<_> = state.field.iter().copied().collect();
    let my_collect_cards: HashSet<_> = state.pile[turn_player].iter().copied().collect();
    let op_collect_cards: HashSet<_> = state.pile[idle_player].iter().copied().collect();
    let mut unseen_cards: HashSet<_> = state.hand[idle_player].iter().copied().collect();
    for card in &state.stock {
        unseen_cards.insert(*card);
    };
    let mut card_state = vec!();
    for cards in CARD_LIST.iter() {
        card_state.push(inter_len(cards, &my_hand_cards) as f32);
    }
    for cards in CARD_LIST.iter() {
        card_state.push(inter_len(cards, &board_cards) as f32);
    }
    for cards in CARD_LIST.iter() {
        card_state.push(inter_len(cards, &my_collect_cards) as f32);
    }
    for cards in CARD_LIST.iter() {
        card_state.push(inter_len(cards, &op_collect_cards) as f32);
    }
    for cards in CARD_LIST.iter() {
        card_state.push(inter_len(cards, &unseen_cards) as f32);
    }

    let card_state: Array1<f32> = Array1::from(card_state);
    let card_state: ArrayView2<f32> = card_state.broadcast((48, card_state.len())).unwrap();
    let card_state: Array2<f32>  = card_state.t().to_owned();

    let mut card_key: Vec<_> = Vec::new();
    for cards in  CARD_LIST.iter() {
        for v in card_to_multi_hot(cards) {
            card_key.push(v);
        }
    }
    let card_key = Array2::from_shape_vec((CARD_LIST.len(), 48), card_key).unwrap();
    ndarray::concatenate![Axis(0), card_state, card_key]
}

pub fn suit_array() -> Array2<f32> {
    let mut array = Array::zeros((12,48));
    for i in 0..12 {
        array.slice_mut(s![i, 4*i..4*i+4]).fill(1.0);
    }
    array
}

fn init_position_array(state: &GameState) -> Array2<f32> {
    let turn_player = state.turn_player();
    let cards_in_my_hand = card_to_multi_hot(&state.hand[turn_player]);
    let cards_in_board = card_to_multi_hot(&state.init_board);
    let unseen_cards = card_to_multi_hot(&state.unseen_cards(turn_player));
    // todo
    ndarray::stack!(Axis(0), cards_in_my_hand, cards_in_board, unseen_cards)
}

fn current_position_array(state: &GameState) -> Array2<f32> {
    let turn_player = state.turn_player();
    let cards_in_my_hand = card_to_multi_hot(&state.hand[turn_player]);
    let cards_in_my_collect = card_to_multi_hot(&state.pile[turn_player]);
    let cards_in_board = card_to_multi_hot(&state.field);
    // Bug Confirmed, for supporting the trained models, keep it as is
    // f_dict['CardInOpCollect'] = card_to_multi_hot(self.pile[self.idle_player])
    let cards_in_op_collect = card_to_multi_hot(&state.pile[turn_player]);
    let unseen_cards = card_to_multi_hot(&state.unseen_cards(turn_player));
    ndarray::stack!(
        Axis(0),
        cards_in_my_hand,
        cards_in_my_collect,
        cards_in_board,
        cards_in_op_collect,
        unseen_cards
    )
}

fn pairing_state_array(state: &GameState) -> Array2<f32> {
    let (showed_cards, paired_cards) = 
        if state.phase == Phase::DiscardPick || state.phase == Phase::DrawPick {
            (card_to_multi_hot(&state.show), card_to_multi_hot(&state.pairing_cards()))
        } else {
            (card_to_multi_hot(&[]), card_to_multi_hot(&[]))
        };
    ndarray::stack!(Axis(0), showed_cards, paired_cards)
}

fn log_array(state: &GameState) -> Array2<f32> {
    let mut turn_list: Vec<_> = (0..state.turn_16).rev().collect();
    for i in state.turn_16..16 {
        turn_list.push(i);
    }
    let mut arr = vec!();
    for i in turn_list {
        // todo: maybe i -1
        for f in &state.card_log[i] {
            arr.push(f.arr);
        }
    }

    Array2::from(arr)
}

pub fn feature_tensor<B: Backend>(state: &GameState, device: &Device<B>) -> Tensor<B, 3> {
    let f = ndarray::concatenate![
        Axis(0),
        reserve_array(),
        game_status_array(state),
        yaku_status_array(state),
        suit_array(),
        init_position_array(state),
        current_position_array(state),
        pairing_state_array(state),
        log_array(state)
    ];
    let (dimx, dimy) = f.dim();

    /*
    println!("verifying");
    let f2: Array2<f64> = read_npy("tensors/test.npy").unwrap();
    for i in 0..300 {
        for j in 0..48 {
            if (f[(i, j)] as f32 - f2[(i, j)] as f32).abs() >= 0.00001 {
                println!("{i} {j} {} {}", f[(i, j)], f2[(i, j)]);
            }
        }
    }
    */

    let flat_arr: Vec<f32> = f
        .outer_iter().flat_map(|row| row.to_vec())
        .collect();

    Tensor::<B,1>::from_data(flat_arr.as_slice(), &device)
        .reshape([1, dimx, dimy])
}

pub fn action_mask(state: &GameState) -> Vec<f32> {
    match state.phase {
        Phase::Discard => card_to_multi_hot(&state.hand[state.turn_player()]).to_vec(),
        Phase::DiscardPick | Phase::DrawPick => card_to_multi_hot(&state.pairing_cards()).to_vec(),
        Phase::KoiKoi => vec!(1.0, 1.0),
        _ => vec!()
    }
}