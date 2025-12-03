use anyhow::Result;
use cudarc::driver::CudaContext;
use gnss_dsp::acquisition::{AcquisitionConfiguration, CuFFTAcquisition};
use std::time::Instant;

pub fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mut acquisition = CuFFTAcquisition::new(ctx, stream, &AcquisitionConfiguration::default())?;
    let prns = (1..=4).collect::<Vec<_>>();
    let signal = vec![num_complex::Complex32::default(); 8_000 * 300];
    let now = Instant::now();
    acquisition.set_signal(&signal)?;
    acquisition.acquire(&prns)?;
    let elapsed = now.elapsed();
    println!("elapsed = {elapsed:?}");

    Ok(())
}
