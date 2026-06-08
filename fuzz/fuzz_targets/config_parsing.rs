#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(config) = toml::from_str::<axon::config::Config>(s) {
        let _ = config.validate();
    }
});
