use std::sync::Mutex;
use burn::prelude::*;
use burn::backend::candle::{Candle, CandleDevice};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};

pub mod game;
pub mod game_tensor;
pub mod model;

use game::{GameState, Phase};
use game_tensor::{action_mask, feature_tensor};
use model::{DiscardModel, PickModel, KoiKoiModel};

type B = Candle<f32, i64>;

struct AppState {
    device: Device<B>,
    discard_model: DiscardModel<B>,
    pick_model: PickModel<B>,
    koikoi_model: KoiKoiModel<B>
}

impl AppState {
    fn new() -> Self {
        let device = CandleDevice::default();
        
        let load_args = LoadArgs::new("./tensors/discard_sl.pt".into());
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load(load_args, &device)
            .expect("Should decode state successfully");
        let discard_model = DiscardModel::new(&device).load_record(record);

        let load_args = LoadArgs::new("./tensors/pick_sl.pt".into());
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load(load_args, &device)
            .expect("Should decode state successfully");
        let pick_model = PickModel::new(&device).load_record(record);

        let load_args = LoadArgs::new("./tensors/koikoi_sl.pt".into());
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load(load_args, &device)
            .expect("Should decode state successfully");
        let koikoi_model = KoiKoiModel::new(&device).load_record(record);

        Self {device, discard_model, pick_model, koikoi_model}
    }
}

#[tauri::command]
async fn koikoi_ai(state: tauri::State<'_, Mutex<AppState>>, game_state: GameState) -> Result<usize, ()> {
    let state = state.lock().unwrap();
    let features = feature_tensor(&game_state, &state.device);
    let mask = action_mask(&game_state);
    let mask = Tensor::<B,1>::from_data(mask.as_slice(), &state.device);
    let output = match game_state.phase {
        Phase::Discard => state.discard_model.forward(features),
        Phase::DiscardPick | Phase::DrawPick => state.pick_model.forward(features),
        Phase::KoiKoi => state.koikoi_model.forward(features),
    };
    let output = (output.squeeze(0) / 10).exp() * mask;
    println!("{output}");
    let output = output.argmax(0);
    Ok(output.into_scalar() as usize)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(AppState::new()))
        .invoke_handler(tauri::generate_handler![koikoi_ai])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
