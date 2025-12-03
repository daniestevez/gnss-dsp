use crate::{
    cufft::CuFFTPlan,
    dsp::{bpsk_modulate, frequency_shift, real_to_complex},
};
use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, safe::CompileOptions},
};
use ndarray::prelude::*;
use num_complex::Complex32;
use std::{ops::RangeInclusive, sync::Arc};

#[derive(Debug)]
pub struct CuFFTAcquisition {
    stream: Arc<CudaStream>,
    parameters: Parameters,
    buffers: Buffers,
    plans: Plans,
    kernels: Kernels,
}

/// Acquisition configuration.
///
/// This struct contains the configuration of the acquisition parameters. It is
/// used to construct an acquisition object.
#[derive(Debug, Clone, PartialEq)]
pub struct AcquisitionConfiguration {
    /// Sample rate in samples per second.
    ///
    /// The sample rate must be an integer.
    pub sample_rate: u32,
    /// Coherent integration time in ms.
    pub coherent_integration_ms: usize,
    /// Number of coherent integrations.
    pub num_noncoherent_integrations: usize,
    /// Doppler range.
    ///
    /// This is the required Doppler range to cover. The actual Doppler range is
    /// incremented according to the Doppler grid.
    pub doppler_range: RangeInclusive<f64>,
    /// Doppler oversampling.
    ///
    /// Number of Doppler bins to use per FFT bin.
    pub doppler_oversampling: usize,
    /// Doppler block size.
    ///
    /// Number of FFT Doppler bins to evaluate in parallel. This parameter
    /// affects GPU memory usage.
    pub doppler_block_size: usize,
}

impl Default for AcquisitionConfiguration {
    fn default() -> AcquisitionConfiguration {
        AcquisitionConfiguration {
            sample_rate: 8_000_000,
            coherent_integration_ms: 20,
            num_noncoherent_integrations: 14,
            doppler_range: -10e3..=10e3,
            doppler_oversampling: 2,
            doppler_block_size: 8,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct Parameters {
    threads_per_block: usize,
    samples_per_ms: usize,
    coherent_ms: usize,
    num_noncoherent: usize,
    doppler_oversampling: usize,
    doppler_block_size: usize,
    num_doppler_blocks: usize,
    first_doppler_bin: isize,
}

#[derive(Debug)]
struct Buffers {
    signal_fft_cuda: CudaSlice<f32>,
    signal_fft_cuda_is_set: bool,
    replicas_fft_cuda: CudaSlice<f32>,
    products_cuda: CudaSlice<f32>,
    iffts_cuda: CudaSlice<f32>,
    results_cuda: CudaSlice<f32>,
}

#[derive(Debug)]
struct Plans {
    signal_plan: CuFFTPlan,
    replicas_plan: CuFFTPlan,
    iffts_plan: CuFFTPlan,
}

#[derive(Debug)]
struct Kernels {
    signal_replica_products: CudaFunction,
    noncoherent_accumulate: CudaFunction,
}

impl CuFFTAcquisition {
    pub fn new(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        configuration: &AcquisitionConfiguration,
    ) -> Result<CuFFTAcquisition> {
        let parameters = Parameters::new(configuration)?;
        let kernels = Kernels::new(&ctx, &parameters)?;
        let plans = Plans::new(&stream, &parameters)?;
        // SAFETY: any use of buffers will overwrite uninitialized buffers with
        // good data
        let buffers = unsafe { Buffers::new(&stream, &parameters)? };

        Ok(CuFFTAcquisition {
            stream,
            parameters,
            plans,
            buffers,
            kernels,
        })
    }

    pub fn set_signal(&mut self, signal: &[Complex32]) -> Result<()> {
        self.compute_signal_fft(signal)?;
        Ok(())
    }

    pub fn acquire(&mut self, prns: &[usize]) -> Result<Array3<f32>> {
        let mut results = Array3::zeros((
            prns.len(),
            self.parameters.results_dopplers(),
            self.parameters.coherent_samples(),
        ));
        for (prn_idx, &prn) in prns.iter().enumerate() {
            self.compute_replicas_fft(prn)?;
            // SAFETY: duplicate_replicas_fft is called after compute_replicas_fft,
            // which initializes its input
            unsafe { self.duplicate_replicas_fft() }?;
            for doppler_block in 0..self.parameters.num_doppler_blocks {
                let doppler_bin0 = self.parameters.first_doppler_bin
                    + isize::try_from(doppler_block * self.parameters.doppler_block_size).unwrap();
                let replica_offset = if doppler_bin0 >= 0 {
                    self.parameters.nfft() - usize::try_from(doppler_bin0).unwrap()
                } else {
                    usize::try_from(-doppler_bin0).unwrap()
                };
                // SAFETY: compute_products is called after duplicate_replicas_fft,
                // which initializes its input
                unsafe { self.compute_products(replica_offset) }?;
                // SAFETY: compute_iffts is called after compute_products, which
                // initializes its input
                unsafe { self.compute_iffts() }?;
                let doppler_bin0_hz = doppler_bin0 as f64 * self.parameters.doppler_bin_hz();
                let carrier_frequency = 1575.42e6; // GPS L1 CA
                let offset_per_noncoherent = -doppler_bin0_hz / carrier_frequency
                    * self.parameters.coherent_samples() as f64;
                // SAFETY: compute_noncoherent_accumulation is called after
                // compute_iffts, which initializes its input
                unsafe {
                    self.compute_noncoherent_accumulation(doppler_block, offset_per_noncoherent)
                }?;
            }
            let mut results_slice = results.slice_mut(s![prn_idx, .., ..]);
            self.stream.memcpy_dtoh(
                &self.buffers.results_cuda,
                results_slice.as_slice_mut().unwrap(),
            )?;
        }
        Ok(results)
    }

    pub fn doppler_axis(&self) -> impl Iterator<Item = f64> {
        self.parameters.doppler_axis()
    }

    pub fn sample_rate(&self) -> f64 {
        self.parameters.samp_rate()
    }

    fn compute_signal_fft(&mut self, signal: &[Complex32]) -> Result<()> {
        let num_signal_samples = self.parameters.num_signal_samples();
        anyhow::ensure!(
            signal.len() >= num_signal_samples,
            "not enough signal samples; required {num_signal_samples} got {}",
            signal.len()
        );

        let signal_cuda = self
            .stream
            .clone_htod(complex32_slice_to_f32(&signal[..num_signal_samples]))?;
        self.plans.signal_plan.execute_c2c_forward(
            signal_cuda.as_view(),
            self.buffers.signal_fft_cuda.as_view_mut(),
        )?;
        self.buffers.signal_fft_cuda_is_set = true;
        Ok(())
    }

    fn compute_replicas_fft(&mut self, prn: usize) -> Result<()> {
        let code = crate::gps::l1ca::ca_code(prn)?;
        let mut replicas = vec![Complex32::new(0.0, 0.0); self.parameters.num_replica_samples()];
        let chip_rate = 1.023e6;
        let replica_zero_doppler = real_to_complex(bpsk_modulate(
            &code,
            chip_rate,
            (self.parameters.samples_per_ms * 1000) as f64,
        ))
        .take(self.parameters.coherent_samples())
        .collect::<Vec<_>>();
        for (subdoppler_bin, replica) in replicas
            .chunks_exact_mut(self.parameters.nfft())
            .enumerate()
        {
            let subdoppler = isize::try_from(subdoppler_bin).unwrap()
                - isize::try_from(self.parameters.doppler_oversampling).unwrap() / 2;
            if subdoppler != 0 {
                let doppler_cycles_sample = subdoppler as f64
                    / (self.parameters.doppler_oversampling * self.parameters.nfft()) as f64;
                let replica_with_doppler = frequency_shift(
                    replica_zero_doppler.iter().copied(),
                    doppler_cycles_sample,
                    1.0,
                );
                for (z, w) in replica.iter_mut().zip(replica_with_doppler) {
                    *z = w;
                }
                // This only writes the first replica_zero_doppler.len() samples of
                // replica. The rest is left as zero padding for the FFT
            } else {
                replica[..replica_zero_doppler.len()].copy_from_slice(&replica_zero_doppler);
            }
        }

        let replicas_cuda = self.stream.clone_htod(complex32_slice_to_f32(&replicas))?;
        self.plans.replicas_plan.execute_c2c_forward(
            replicas_cuda.as_view(),
            self.buffers
                .replicas_fft_cuda
                .slice_mut(..self.buffers.replicas_fft_cuda.len() - 2 * self.parameters.nfft()),
        )?;
        Ok(())
    }

    // SAFETY: this function reads from self.buffers.replicas_fft_cuda, so it
    // requires that it has been partially initialized before, for instance by
    // running self.compute_replicas_fft
    unsafe fn duplicate_replicas_fft(&mut self) -> Result<()> {
        for n in 0..self.parameters.doppler_oversampling {
            // 4x because there is a 2x for Complex<f32> and another 2x because of
            // cyclic duplication
            let offset = 4 * self.parameters.nfft() * n;
            let (a, mut b) = self
                .buffers
                .replicas_fft_cuda
                .split_at_mut(offset + 2 * self.parameters.nfft());
            self.stream.memcpy_dtod(
                &a.slice(offset..),
                &mut b.slice_mut(..2 * self.parameters.nfft()),
            )?;
        }
        Ok(())
    }

    // SAFETY: This function reads from self.buffers.replicas_fft_cuda, so it
    // requires that it has been fully initialized before, for instance by
    // running self.compute_replicas_fft and
    // self.duplicate_replicas_fft.
    //
    // Additionally this function reads from self.buffers.signal_fft_cuda, so it
    // checks self.buffers.signal_fft_cuda_is_set to ensure that
    // self.compute_signal_fft has been called before and
    // self.buffers.signal_fft_cuda is initialized.
    unsafe fn compute_products(&mut self, replica_offset: usize) -> Result<()> {
        anyhow::ensure!(
            self.buffers.signal_fft_cuda_is_set,
            "set_signal must be called before acquire"
        );
        let mut builder = self
            .stream
            .launch_builder(&self.kernels.signal_replica_products);
        builder.arg(&self.buffers.signal_fft_cuda);
        let replicas_slice = self.buffers.replicas_fft_cuda.slice(2 * replica_offset..);
        builder.arg(&replicas_slice);
        builder.arg(&mut self.buffers.products_cuda);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    self.parameters.block_x(),
                    self.parameters.num_noncoherent.try_into().unwrap(),
                    self.parameters.doppler_block_size.try_into().unwrap(),
                ),
                block_dim: self.parameters.block_dim(),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    // SAFETY: this function reads from self.buffers.products_cuda, so it
    // requires that it has been initialized before, for instance by running
    // self.compute_products.
    unsafe fn compute_iffts(&mut self) -> Result<()> {
        self.plans.iffts_plan.execute_c2c_inverse(
            self.buffers.products_cuda.as_view(),
            self.buffers.iffts_cuda.as_view_mut(),
        )?;
        Ok(())
    }

    // SAFETY: this function reads from self.buffers.iffts_cuda, so it requires
    // that it has been initialized before, for instance by running
    // self.compute_iffts
    unsafe fn compute_noncoherent_accumulation(
        &mut self,
        doppler_block: usize,
        offset_per_noncoherent: f64,
    ) -> Result<()> {
        let mut builder = self
            .stream
            .launch_builder(&self.kernels.noncoherent_accumulate);
        builder.arg(&self.buffers.iffts_cuda);
        let mut results_slice = self.buffers.results_cuda.slice_mut(
            self.parameters.results_block_size() * doppler_block
                ..self.parameters.results_block_size() * (doppler_block + 1),
        );
        let offset_per_noncoherent = offset_per_noncoherent as f32;
        builder.arg(&offset_per_noncoherent);
        builder.arg(&mut results_slice);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    self.parameters
                        .coherent_samples()
                        .div_ceil(self.parameters.threads_per_block)
                        .try_into()
                        .unwrap(),
                    (self.parameters.doppler_oversampling * self.parameters.doppler_block_size)
                        .try_into()
                        .unwrap(),
                    1,
                ),
                block_dim: self.parameters.block_dim(),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }
}

impl Parameters {
    fn new(configuration: &AcquisitionConfiguration) -> Result<Parameters> {
        anyhow::ensure!(
            configuration.sample_rate.is_multiple_of(1000),
            "there must be an integer number of samples per millisecond"
        );
        anyhow::ensure!(
            configuration.coherent_integration_ms > 0,
            "the coherent integration length must not be zero"
        );
        anyhow::ensure!(
            configuration.num_noncoherent_integrations > 0,
            "the number of non-coherent integrations must not be zero"
        );
        anyhow::ensure!(
            configuration.doppler_oversampling > 0,
            "the Doppler oversampling must not be zero"
        );
        anyhow::ensure!(
            configuration.doppler_block_size > 0,
            "the Doppler block size must not be zero"
        );

        let samples_per_ms = (configuration.sample_rate / 1000).try_into().unwrap();

        let nfft = 2 * configuration.coherent_integration_ms * samples_per_ms;
        let doppler_bin_width = configuration.sample_rate as f64 / nfft as f64;
        let min_subdoppler = -isize::try_from(configuration.doppler_oversampling).unwrap() / 2;
        let min_subdoppler = min_subdoppler as f64 * configuration.sample_rate as f64
            / (configuration.doppler_oversampling * nfft) as f64;
        let first_doppler_bin = ((*configuration.doppler_range.start() + min_subdoppler)
            / doppler_bin_width)
            .floor() as isize;
        // The first Doppler bin must be a multiple of the Doppler block size
        let first_doppler_bin = first_doppler_bin
            .div_euclid(isize::try_from(configuration.doppler_oversampling).unwrap())
            * isize::try_from(configuration.doppler_oversampling).unwrap();
        let max_subdoppler =
            configuration.doppler_oversampling - 1 - configuration.doppler_oversampling / 2;
        let max_subdoppler = max_subdoppler as f64 * configuration.sample_rate as f64
            / (configuration.doppler_oversampling * nfft) as f64;
        let last_doppler_bin = ((*configuration.doppler_range.end() - max_subdoppler)
            / doppler_bin_width)
            .ceil() as isize;
        let total_doppler_bins = usize::try_from(last_doppler_bin - first_doppler_bin + 1).unwrap();
        let adjusted_doppler_bins =
            total_doppler_bins.next_multiple_of(configuration.doppler_block_size);
        let num_doppler_blocks = adjusted_doppler_bins / configuration.doppler_block_size;

        let parameters = Parameters {
            threads_per_block: 512,
            samples_per_ms,
            coherent_ms: configuration.coherent_integration_ms,
            num_noncoherent: configuration.num_noncoherent_integrations,
            doppler_oversampling: configuration.doppler_oversampling,
            doppler_block_size: configuration.doppler_block_size,
            num_doppler_blocks,
            first_doppler_bin,
        };
        assert_eq!(parameters.nfft(), nfft);
        Ok(parameters)
    }

    fn coherent_samples(&self) -> usize {
        self.coherent_ms * self.samples_per_ms
    }

    fn nfft(&self) -> usize {
        2 * self.coherent_samples()
    }

    fn num_signal_samples(&self) -> usize {
        (self.num_noncoherent - 1) * self.coherent_samples() + self.nfft()
    }

    fn num_replica_samples(&self) -> usize {
        self.nfft() * self.doppler_oversampling
    }

    fn num_product_samples(&self) -> usize {
        self.nfft() * self.results_block_dopplers() * self.num_noncoherent
    }

    fn results_block_size(&self) -> usize {
        self.coherent_samples() * self.results_block_dopplers()
    }

    fn results_block_dopplers(&self) -> usize {
        self.doppler_oversampling * self.doppler_block_size
    }

    fn results_samples(&self) -> usize {
        self.results_block_size() * self.num_doppler_blocks
    }

    fn results_dopplers(&self) -> usize {
        self.results_block_dopplers() * self.num_doppler_blocks
    }

    fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block.try_into().unwrap(), 1, 1)
    }

    fn block_x(&self) -> u32 {
        self.nfft()
            .div_ceil(self.threads_per_block)
            .try_into()
            .unwrap()
    }

    fn samp_rate(&self) -> f64 {
        (self.samples_per_ms * 1000) as f64
    }

    fn doppler_bin_hz(&self) -> f64 {
        self.samp_rate() / self.nfft() as f64
    }

    fn doppler_axis(&self) -> impl Iterator<Item = f64> {
        (self.first_doppler_bin..)
            .take(self.num_doppler_blocks * self.doppler_block_size)
            .flat_map(move |bin| {
                let freq = bin as f64 * self.doppler_bin_hz();
                (0..self.doppler_oversampling).map(move |subdoppler| {
                    let subdoppler = isize::try_from(subdoppler).unwrap()
                        - isize::try_from(self.doppler_oversampling).unwrap() / 2;
                    let subdoppler = self.samp_rate() * subdoppler as f64
                        / (self.doppler_oversampling * self.nfft()) as f64;
                    freq + subdoppler
                })
            })
    }
}

impl Buffers {
    // SAFETY: all the buffers created by this function are uninitialized
    unsafe fn new(stream: &Arc<CudaStream>, parameters: &Parameters) -> Result<Buffers> {
        let signal_fft_cuda =
            unsafe { stream.alloc::<f32>(2 * parameters.nfft() * parameters.num_noncoherent) }?;
        // 4x because there is a 2x for Complex<f32> and another 2x because of
        // cyclic duplication
        let replicas_fft_cuda =
            unsafe { stream.alloc::<f32>(4 * parameters.num_replica_samples()) }?;
        let products_cuda = unsafe { stream.alloc::<f32>(2 * parameters.num_product_samples()) }?;
        let iffts_cuda = unsafe { stream.alloc::<f32>(2 * parameters.num_product_samples()) }?;
        let results_cuda = unsafe { stream.alloc::<f32>(parameters.results_samples()) }?;
        Ok(Buffers {
            signal_fft_cuda,
            signal_fft_cuda_is_set: false,
            replicas_fft_cuda,
            products_cuda,
            iffts_cuda,
            results_cuda,
        })
    }
}

impl Plans {
    fn new(stream: &Arc<CudaStream>, parameters: &Parameters) -> Result<Plans> {
        let signal_plan = CuFFTPlan::new_c2c_batch_with_idist(
            stream,
            parameters.nfft(),
            parameters.num_noncoherent,
            parameters.coherent_samples(),
        )?;
        let replicas_plan = CuFFTPlan::new_c2c_batch_with_odist(
            stream,
            parameters.nfft(),
            parameters.doppler_oversampling,
            2 * parameters.nfft(),
        )?;
        let iffts_plan = CuFFTPlan::new_c2c_batch(
            stream,
            parameters.nfft(),
            parameters.doppler_block_size
                * parameters.doppler_oversampling
                * parameters.num_noncoherent,
        )?;
        Ok(Plans {
            signal_plan,
            replicas_plan,
            iffts_plan,
        })
    }
}

impl Kernels {
    fn new(ctx: &Arc<CudaContext>, parameters: &Parameters) -> Result<Kernels> {
        let ptx = compile_ptx_with_opts(
            kernels_ptx(parameters),
            CompileOptions {
                // TODO: do not hardcode include path
                include_paths: vec!["/opt/cuda/targets/x86_64-linux/include/".to_string()],
                ..Default::default()
            },
        )?;
        let module = ctx.load_module(ptx)?;
        let signal_replica_products = module.load_function("signal_replica_products")?;
        let noncoherent_accumulate = module.load_function("noncoherent_accumulate")?;
        Ok(Kernels {
            signal_replica_products,
            noncoherent_accumulate,
        })
    }
}

fn kernels_ptx(parameters: &Parameters) -> String {
    let nfft = parameters.nfft();
    let coherent_samples = parameters.coherent_samples();
    let doppler_oversampling = parameters.doppler_oversampling;
    let num_noncoherent = parameters.num_noncoherent;
    format!(
        r#"
#include <cuComplex.h>

extern "C" __global__ void signal_replica_products(
        const cuComplex* signal,
        const cuComplex* replicas,
        cuComplex* products
    ) {{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int signal_offset = blockIdx.y * {nfft};
    const int replica_idx = blockIdx.z;
    for (int i = 0; i < {doppler_oversampling}; ++i) {{
        for (int j = index; j < {nfft}; j += stride) {{
            const auto z = signal[signal_offset + j];
            const auto w = replicas[i * 2 * {nfft} + j - replica_idx];
            const auto prod = cuCmulf(z, cuConjf(w));
            const int out_idx = (replica_idx * {doppler_oversampling} + i) * {nfft} * {num_noncoherent}
                + signal_offset + j;
            products[out_idx] = prod;
        }}
    }}
}}

extern "C" __global__ void noncoherent_accumulate(
        const cuComplex* z,
        float offset_per_noncoherent,
        float* out
    ) {{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int z_offset = blockIdx.y * {nfft} * {num_noncoherent};
    const int out_offset = blockIdx.y * {coherent_samples};
    for (int i = index; i < {coherent_samples}; i += stride) {{
        float accumulator = 0.0f;
        for (int j = 0; j < {num_noncoherent}; ++j) {{
            const int offset = roundf(offset_per_noncoherent * j);
            auto i_corrected = i + offset;
            if (i_corrected < 0) {{
                i_corrected -= {coherent_samples};
            }} else if (i_corrected >= {coherent_samples}
                    && j * {nfft} + i_corrected + {coherent_samples} < {nfft} * {num_noncoherent}) {{
                i_corrected += {coherent_samples};
            }}
            const auto a = z[z_offset + j * {nfft} + i_corrected];
            const float magsq = cuCrealf(a) * cuCrealf(a) + cuCimagf(a) * cuCimagf(a);
            accumulator += magsq;
        }}
        out[out_offset + i] = accumulator;
    }}
}}
"#
    )
}

fn complex32_slice_to_f32(x: &[Complex32]) -> &[f32] {
    // SAFETY: the memory layout of [Complex32] is compatible with a [f32] with
    // twice as many elements
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<f32>(), 2 * x.len()) }
}

#[cfg(test)]
mod test {
    use super::*;
    use num_complex::ComplexFloat;
    use rand::prelude::*;
    use std::sync::Mutex;

    // Mutex to avoid having tests running simultaneously on the GPU, since that
    // can cause out-of-memory on the GPU allocations
    static GPU_MUTEX: Mutex<()> = Mutex::new(());

    fn f32_slice_to_complex32(x: &[f32]) -> &[Complex32] {
        assert!(x.len().is_multiple_of(2));
        // SAFETY: the memory layout of [Complex32] is compatible with a [f32] with
        // twice as many elements
        unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<Complex32>(), x.len() / 2) }
    }

    fn new_complex32_random(rng: &mut ThreadRng, n: usize) -> Vec<Complex32> {
        let range = -1.0..=1.0;
        std::iter::repeat_with(|| {
            Complex32::new(
                rng.random_range(range.clone()),
                rng.random_range(range.clone()),
            )
        })
        .take(n)
        .collect()
    }

    macro_rules! assert_close {
        ($x:expr, $y:expr, $relative_tolerance:expr) => {
            assert_close!($x, $y, $relative_tolerance, 0.0);
        };
        ($x:expr, $y:expr, $relative_tolerance:expr, $absolute_tolerance:expr) => {
            assert_eq!($x.len(), $y.len());
            for (a, b) in $x.iter().zip($y.iter()) {
                assert!(
                    (2.0 * (a - b).abs() / (a.abs() + b.abs()) <= $relative_tolerance)
                        || (a - b).abs() <= $absolute_tolerance,
                    "{a} and {b} differ by more than relative tolerance {} and absolute tolerance {}",
                    $relative_tolerance,
                    $absolute_tolerance,
                );
            }
        };
    }

    #[test]
    fn compute_signal_fft() {
        let _gpu = GPU_MUTEX.lock().unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut acquisition =
            CuFFTAcquisition::new(ctx, stream.clone(), &AcquisitionConfiguration::default())
                .unwrap();
        let mut rng = rand::rng();
        let signal = new_complex32_random(&mut rng, acquisition.parameters.num_signal_samples());
        acquisition.compute_signal_fft(&signal).unwrap();
        let signal_fft = stream
            .clone_dtoh(&acquisition.buffers.signal_fft_cuda)
            .unwrap();

        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(acquisition.parameters.nfft());
        let signal_rustfft = signal
            .windows(acquisition.parameters.nfft())
            .step_by(acquisition.parameters.coherent_samples())
            .flat_map(|window| {
                let mut window = window.to_vec();
                fft.process(&mut window);
                window
            })
            .collect::<Vec<_>>();

        assert_close!(f32_slice_to_complex32(&signal_fft), &signal_rustfft, 5e-3);
    }

    #[test]
    fn compute_replicas_fft() {
        let _gpu = GPU_MUTEX.lock().unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut acquisition =
            CuFFTAcquisition::new(ctx, stream.clone(), &AcquisitionConfiguration::default())
                .unwrap();
        let prn = 1;
        acquisition.compute_replicas_fft(prn).unwrap();

        let replicas_fft = stream
            .clone_dtoh(&acquisition.buffers.replicas_fft_cuda)
            .unwrap();
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(acquisition.parameters.nfft());
        // step_by(2) because replicas_fft has room to duplicate the replicas
        // FFTs using duplicate_replicas_fft
        for (bin, replica_fft) in f32_slice_to_complex32(&replicas_fft)
            .chunks_exact(acquisition.parameters.nfft())
            .step_by(2)
            .enumerate()
        {
            let code = crate::gps::l1ca::ca_code(prn).unwrap();
            let samp_rate = 8e6;
            let bin = isize::try_from(bin).unwrap()
                - isize::try_from(acquisition.parameters.doppler_oversampling).unwrap() / 2;
            let doppler = samp_rate * (bin as f64)
                / (acquisition.parameters.doppler_oversampling * acquisition.parameters.nfft())
                    as f64;
            let chip_rate = 1.023e6;
            let samp_rate = 8e6;
            let mut replica_rustfft = frequency_shift(
                real_to_complex(bpsk_modulate(&code, chip_rate, samp_rate)),
                doppler,
                samp_rate,
            )
            .take(acquisition.parameters.coherent_samples())
            .chain(std::iter::repeat(Complex32::new(0.0, 0.0)))
            .take(acquisition.parameters.nfft())
            .collect::<Vec<_>>();
            fft.process(&mut replica_rustfft);
            assert_close!(replica_fft, &replica_rustfft, 3e-3, 1e-3);
        }
    }

    #[test]
    fn compute_products() {
        let _gpu = GPU_MUTEX.lock().unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut acquisition =
            CuFFTAcquisition::new(ctx, stream.clone(), &AcquisitionConfiguration::default())
                .unwrap();
        let mut rng = rand::rng();
        let signal_fft = new_complex32_random(
            &mut rng,
            acquisition.parameters.nfft() * acquisition.parameters.num_noncoherent,
        );
        stream
            .memcpy_htod(
                complex32_slice_to_f32(&signal_fft),
                &mut acquisition.buffers.signal_fft_cuda,
            )
            .unwrap();
        acquisition.buffers.signal_fft_cuda_is_set = true;
        let replicas_fft =
            new_complex32_random(&mut rng, acquisition.parameters.num_replica_samples());
        let mut replicas_fft_2x = Vec::new();
        for replica in replicas_fft.chunks_exact(acquisition.parameters.nfft()) {
            replicas_fft_2x.extend_from_slice(replica);
            replicas_fft_2x.resize(
                replicas_fft_2x.len() + acquisition.parameters.nfft(),
                Complex32::new(0.0, 0.0),
            );
        }
        stream
            .memcpy_htod(
                complex32_slice_to_f32(&replicas_fft_2x),
                &mut acquisition.buffers.replicas_fft_cuda,
            )
            .unwrap();

        let replica_offset = rng.random_range(
            0..=acquisition.parameters.nfft() - acquisition.parameters.doppler_block_size,
        );
        // SAFETY: the input buffer replicas_fft_cuda has been initalized with
        // test data
        unsafe {
            acquisition.duplicate_replicas_fft().unwrap();
            acquisition.compute_products(replica_offset).unwrap();
        }
        let products = stream
            .clone_dtoh(&acquisition.buffers.products_cuda)
            .unwrap();

        let mut products_expected = Vec::new();
        for doppler_idx in 0..acquisition.parameters.doppler_block_size {
            for replica in replicas_fft.chunks_exact(acquisition.parameters.nfft()) {
                for signal in signal_fft.chunks_exact(acquisition.parameters.nfft()) {
                    for (z, w) in signal.iter().zip(
                        replica
                            .iter()
                            .cycle()
                            .skip(replica_offset + acquisition.parameters.nfft() - doppler_idx),
                    ) {
                        products_expected.push(z * w.conj());
                    }
                }
            }
        }

        assert_close!(f32_slice_to_complex32(&products), &products_expected, 1e-6);
    }

    #[test]
    fn compute_iffts() {
        let _gpu = GPU_MUTEX.lock().unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut acquisition =
            CuFFTAcquisition::new(ctx, stream.clone(), &AcquisitionConfiguration::default())
                .unwrap();
        let mut rng = rand::rng();
        let products = new_complex32_random(&mut rng, acquisition.parameters.num_product_samples());
        stream
            .memcpy_htod(
                complex32_slice_to_f32(&products),
                &mut acquisition.buffers.products_cuda,
            )
            .unwrap();

        // SAFETY: the required input buffer products_cuda has been initialized
        // with test data
        unsafe { acquisition.compute_iffts().unwrap() };

        let iffts = stream.clone_dtoh(&acquisition.buffers.iffts_cuda).unwrap();

        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_inverse(acquisition.parameters.nfft());
        let mut iffts_rustfft = products.clone();
        fft.process(&mut iffts_rustfft);

        assert_close!(f32_slice_to_complex32(&iffts), &iffts_rustfft, 2e-2);
    }

    #[test]
    fn compute_noncoherent_accumulation() {
        let _gpu = GPU_MUTEX.lock().unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut acquisition =
            CuFFTAcquisition::new(ctx, stream.clone(), &AcquisitionConfiguration::default())
                .unwrap();
        let mut rng = rand::rng();
        let iffts = new_complex32_random(&mut rng, acquisition.parameters.num_product_samples());
        stream
            .memcpy_htod(
                complex32_slice_to_f32(&iffts),
                &mut acquisition.buffers.iffts_cuda,
            )
            .unwrap();

        let doppler_block = rng.random_range(0..acquisition.parameters.num_doppler_blocks);
        let offset_per_noncoherent: f32 = rng.random_range(-3.0..=3.0);
        // SAFETY: the required input buffer iffts_cuda has been initialized
        // with test data. Additionally, the dtov below only accesses the part
        // of the output buffer that has actually been written by this call.
        unsafe {
            acquisition
                .compute_noncoherent_accumulation(doppler_block, offset_per_noncoherent.into())
                .unwrap()
        };

        let results_offset = doppler_block * acquisition.parameters.results_block_size();
        let results_len = acquisition.parameters.results_block_size();
        let partial_results = stream
            .clone_dtoh(
                &acquisition
                    .buffers
                    .results_cuda
                    .slice(results_offset..results_offset + results_len),
            )
            .unwrap();

        let coherent_samples = acquisition.parameters.coherent_samples();
        let nfft = acquisition.parameters.nfft();
        let results_expected = iffts
            .chunks_exact(nfft * acquisition.parameters.num_noncoherent)
            .flat_map(|chunk| {
                (0..coherent_samples).map(|idx| {
                    let mut accumulator = 0.0;
                    for fft_idx in 0..acquisition.parameters.num_noncoherent {
                        let offset = (offset_per_noncoherent * fft_idx as f32).round() as isize;
                        let idx = isize::try_from(idx).unwrap() + offset;
                        let coherent_samples_i = isize::try_from(coherent_samples).unwrap();
                        let (fft_idx, idx) = if idx < 0 {
                            assert!(fft_idx > 0);
                            (fft_idx - 1, idx + coherent_samples_i)
                        } else if idx >= coherent_samples_i
                            && fft_idx + 1 < acquisition.parameters.num_noncoherent
                        {
                            (fft_idx + 1, idx - coherent_samples_i)
                        } else {
                            (fft_idx, idx)
                        };
                        let idx = usize::try_from(idx).unwrap();
                        let z = chunk[fft_idx * nfft + idx];
                        accumulator += z.norm_sqr();
                    }
                    accumulator
                })
            })
            .collect::<Vec<f32>>();

        assert_close!(&partial_results, &results_expected, 1e-6);
    }
}
