#[pyo3::pymodule]
mod gnss_dsp {
    use anyhow::Context;
    use cudarc::driver::CudaContext;
    use num_complex::Complex32;
    use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, ToPyArray};
    use pyo3::prelude::*;

    #[pyclass]
    struct CuFFTAcquisition(crate::acquisition::CuFFTAcquisition);

    #[pymethods]
    impl CuFFTAcquisition {
        #[allow(clippy::too_many_arguments)]
        #[new]
        // TODO: min_doppler's default value shows up as Ellipsis in python
        #[pyo3(signature = (
            sample_rate, *,
            coherent_integration_ms=20,
            num_noncoherent_integrations=14,
            min_doppler=-10e3,
            max_doppler=10e3,
            doppler_oversampling=2,
            doppler_block_size=8,
            cuda_device_ordinal=0
        ))]
        fn new(
            sample_rate: u32,
            coherent_integration_ms: usize,
            num_noncoherent_integrations: usize,
            min_doppler: f64,
            max_doppler: f64,
            doppler_oversampling: usize,
            doppler_block_size: usize,
            cuda_device_ordinal: usize,
        ) -> PyResult<Self> {
            let ctx =
                CudaContext::new(cuda_device_ordinal).context("Failed to create CUDA context")?;
            let stream = ctx.default_stream();
            let configuration = crate::acquisition::AcquisitionConfiguration {
                sample_rate,
                coherent_integration_ms,
                num_noncoherent_integrations,
                doppler_range: min_doppler..=max_doppler,
                doppler_oversampling,
                doppler_block_size,
            };
            let acq = crate::acquisition::CuFFTAcquisition::new(ctx, stream, &configuration)?;
            Ok(Self(acq))
        }

        fn set_signal<'py>(
            &mut self,
            _py: Python<'py>,
            signal: PyReadonlyArray1<'py, Complex32>,
        ) -> PyResult<()> {
            self.0.set_signal(signal.as_slice()?)?;
            Ok(())
        }

        fn acquire<'py>(
            &mut self,
            py: Python<'py>,
            prns: Vec<usize>,
        ) -> PyResult<Bound<'py, PyArray3<f32>>> {
            let results = self.0.acquire(&prns)?;
            Ok(results.into_pyarray(py))
        }

        fn doppler_axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.doppler_axis().collect::<Vec<_>>().into_pyarray(py)
        }

        fn sample_rate(&self) -> f64 {
            self.0.sample_rate()
        }
    }

    #[pyfunction]
    fn gps_l1_ca_code<'py>(py: Python<'py>, prn: usize) -> PyResult<Bound<'py, PyArray1<u8>>> {
        Ok(crate::gps::l1ca::ca_code(prn)?.to_pyarray(py))
    }
}
