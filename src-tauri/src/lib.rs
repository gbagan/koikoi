use std::sync::Mutex;
use burn::prelude::*;
use burn::backend::candle::{Candle, CandleDevice};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};

pub mod game;
pub mod game_tensor;
pub mod model;

use game::GameState;
use game_tensor::feature_tensor;
use model::PickModel;

type B = Candle<f32, i64>;

struct AppState {
    device: Device<B>,
    pick_model: PickModel<B>,
}


impl AppState {
    fn new() -> Self {
        let device = CandleDevice::default();
        let load_args = LoadArgs::new("./tensors/pick_sl.pt".into()); //.with_debug_print();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load(load_args, &device)
            .expect("Should decode state successfully");
        let pick_model: PickModel<B> = PickModel::new(&device).load_record(record);
        Self {device, pick_model}
    }
}

#[tauri::command]
async fn test(state: tauri::State<'_, Mutex<AppState>>, game_state: GameState) -> Result<(), ()> {
    let state = state.lock().unwrap();
    let tensor = feature_tensor(&game_state, &state.device);
    println!("{:?}", tensor.dims());
    state.pick_model.forward(tensor);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(AppState::new()))
        .invoke_handler(tauri::generate_handler![test])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
