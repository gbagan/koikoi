use std::sync::Mutex;
use burn::prelude::*;
use burn::backend::candle::{Candle, CandleDevice};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};

pub mod game;
pub mod game_tensor;
pub mod model;

use game::GameState;
use game_tensor::{action_mask, feature_tensor};
use model::DiscardModel;

use ndarray_npy::read_npy;


type B = Candle<f32, i64>;

struct AppState {
    device: Device<B>,
    discard_model: DiscardModel<B>,
}


impl AppState {
    fn new() -> Self {
        let device = CandleDevice::default();
        let load_args = LoadArgs::new("./tensors/discard_sl.pt".into()); //.with_debug_print();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load(load_args, &device)
            .expect("Should decode state successfully");
        let discard_model: DiscardModel<B> = DiscardModel::new(&device).load_record(record);
        Self {device, discard_model}
    }
}

#[tauri::command]
async fn test(state: tauri::State<'_, Mutex<AppState>>, game_state: GameState) -> Result<(), ()> {
    let state = state.lock().unwrap();
    let features = feature_tensor(&game_state, &state.device);
    //let output = state.pick_model.forward(tensor);
    //let mov = output.argmax(1).into_scalar();
    let mask = action_mask(&game_state.round_state);
    let mask = Tensor::<B,1>::from_data(mask.as_slice(), &state.device);
    predict(&state.discard_model, features, mask, &state.device);
    //println!("{}", mask);
    //println!("{mov}");
    Ok(())
}

fn predict<B: Backend>(pick_model: &DiscardModel<B>, features: Tensor<B, 3>, mask: Tensor<B, 1>, device: &Device<B>) {
    let output = pick_model.forward(features).squeeze(0);
    let output = (output / 10).exp() * mask;
    println!("{output}");
    println!("{}", output.argmax(0));
    //let action_output = self.action_dict[state][output.argmax()];
}


#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(AppState::new()))
        .invoke_handler(tauri::generate_handler![test])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
