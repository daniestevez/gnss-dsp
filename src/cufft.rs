use anyhow::Result;
use cudarc::driver::{CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut};
use std::{ffi::c_int, sync::Arc};

pub mod bindings {
    #![allow(non_camel_case_types)]
    #![allow(non_upper_case_globals)]
    #![allow(non_snake_case)]
    #![allow(unused_imports)]
    #![allow(dead_code)]
    #![allow(clippy::approx_constant)]
    include!(concat!(env!("OUT_DIR"), "/cufft_bindings.rs"));
}

macro_rules! ensure_cufft_success {
    ($ret:expr, $fn: expr) => {
        anyhow::ensure!(
            $ret == crate::cufft::bindings::cufftResult_t_CUFFT_SUCCESS,
            "{} failed: {}",
            $fn,
            cufft_result_to_str($ret),
        )
    };
}

// cufftResult_t values defined in
// https://docs.nvidia.com/cuda/cufft/#return-value-cufftresult
//
// We define these here instead of taking them from the bindings because older
// cufft versions do not define all the possible results in the bindings
//
// ```c
// typedef enum cufftResult_t {
//     CUFFT_SUCCESS            =  0, // The cuFFT operation was successful
//     CUFFT_INVALID_PLAN       =  1, // cuFFT was passed an invalid plan handle
//     CUFFT_ALLOC_FAILED       =  2, // cuFFT failed to allocate GPU or CPU memory
//     CUFFT_INVALID_TYPE       =  3, // The cuFFT type provided is unsupported
//     CUFFT_INVALID_VALUE      =  4, // User specified an invalid pointer or parameter
//     CUFFT_INTERNAL_ERROR     =  5, // Driver or internal cuFFT library error
//     CUFFT_EXEC_FAILED        =  6, // Failed to execute an FFT on the GPU
//     CUFFT_SETUP_FAILED       =  7, // The cuFFT library failed to initialize
//     CUFFT_INVALID_SIZE       =  8, // User specified an invalid transform size
//     CUFFT_UNALIGNED_DATA     =  9, // Not currently in use
//     CUFFT_INVALID_DEVICE     = 11, // Execution of a plan was on different GPU than plan creation
//     CUFFT_NO_WORKSPACE       = 13  // No workspace has been provided prior to plan execution
//     CUFFT_NOT_IMPLEMENTED    = 14, // Function does not implement functionality for parameters given.
//     CUFFT_NOT_SUPPORTED      = 16, // Operation is not supported for parameters given.
//     CUFFT_MISSING_DEPENDENCY = 17, // cuFFT is unable to find a dependency
//     CUFFT_NVRTC_FAILURE      = 18, // An NVRTC failure was encountered during a cuFFT operation
//     CUFFT_NVJITLINK_FAILURE  = 19, // An nvJitLink failure was encountered during a cuFFT operation
//     CUFFT_NVSHMEM_FAILURE    = 20, // An NVSHMEM failure was encountered during a cuFFT operation
// } cufftResult;
// ```

#[repr(u8)]
enum CuFFTResult {
    Success = 0,
    InvalidPlan = 1,
    AllocFailed = 2,
    InvalidType = 3,
    InvalidValue = 4,
    InternalError = 5,
    ExecFailed = 6,
    SetupFailed = 7,
    InvalidSize = 8,
    UnalignedData = 9,
    InvalidDevice = 11,
    NoWorkspace = 13,
    NotImplemented = 14,
    NotSupported = 16,
    MissingDependency = 17,
    NVRTCFailure = 18,
    NVJITLINKFailure = 19,
    NVSHMEMFailure = 20,
}

impl From<CuFFTResult> for bindings::cufftResult_t {
    fn from(value: CuFFTResult) -> bindings::cufftResult_t {
        (value as u8).into()
    }
}

fn cufft_result_to_str(result: bindings::cufftResult_t) -> String {
    match result {
        val if val == bindings::cufftResult_t::from(CuFFTResult::AllocFailed) => {
            "alloc failed".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::ExecFailed) => {
            "exec failed".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::InternalError) => {
            "internal error".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::InvalidDevice) => {
            "invalid device".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::InvalidPlan) => {
            "invalid plan".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::InvalidSize) => {
            "invalid size".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::InvalidType) => {
            "invalid type".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::InvalidValue) => {
            "invalid value".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::MissingDependency) => {
            "missing dependency".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::NotImplemented) => {
            "not implemented".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::NotSupported) => {
            "not supported".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::NoWorkspace) => {
            "no workspace".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::NVJITLINKFailure) => {
            "nvjitlink failure".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::NVRTCFailure) => {
            "nvrtc failure".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::NVSHMEMFailure) => {
            "nvshmem failure".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::SetupFailed) => {
            "setup failed".to_string()
        }
        val if val == bindings::cufftResult_t::from(CuFFTResult::Success) => "success".to_string(),
        val if val == bindings::cufftResult_t::from(CuFFTResult::UnalignedData) => {
            "unaligned data".to_string()
        }
        _ => format!("unknown return code {}", result),
    }
}

#[derive(Debug)]
pub struct CuFFTPlan {
    plan: bindings::cufftHandle,
    stream: Arc<CudaStream>,
    required_input_len: usize,
    required_output_len: usize,
}

impl Drop for CuFFTPlan {
    fn drop(&mut self) {
        unsafe { bindings::cufftDestroy(self.plan) };
    }
}

impl CuFFTPlan {
    pub fn new_c2c_batch(
        stream: &Arc<CudaStream>,
        nfft: usize,
        batch_size: usize,
    ) -> Result<CuFFTPlan> {
        CuFFTPlan::new_c2c_batch_with_idist(stream, nfft, batch_size, nfft)
    }

    pub fn new_c2c_batch_with_idist(
        stream: &Arc<CudaStream>,
        nfft: usize,
        batch_size: usize,
        input_distance: usize,
    ) -> Result<CuFFTPlan> {
        CuFFTPlan::new_c2c_batch_with_idist_and_odist(
            stream,
            nfft,
            batch_size,
            input_distance,
            nfft,
        )
    }

    pub fn new_c2c_batch_with_odist(
        stream: &Arc<CudaStream>,
        nfft: usize,
        batch_size: usize,
        output_distance: usize,
    ) -> Result<CuFFTPlan> {
        CuFFTPlan::new_c2c_batch_with_idist_and_odist(
            stream,
            nfft,
            batch_size,
            nfft,
            output_distance,
        )
    }

    pub fn new_c2c_batch_with_idist_and_odist(
        stream: &Arc<CudaStream>,
        nfft: usize,
        batch_size: usize,
        input_distance: usize,
        output_distance: usize,
    ) -> Result<CuFFTPlan> {
        let mut plan = Default::default();
        let plan_ref = &mut plan;
        let mut n = [c_int::try_from(nfft).unwrap()];
        let n_ref = &mut n[0];
        let required_input_len = if batch_size >= 1 {
            (batch_size - 1) * input_distance + nfft
        } else {
            0
        };
        let mut inembed_array = [c_int::try_from(required_input_len).unwrap()];
        let inembed_array_ref = &mut inembed_array[0];
        let required_output_len = if batch_size >= 1 {
            (batch_size - 1) * output_distance + nfft
        } else {
            0
        };
        let mut onembed_array = [c_int::try_from(required_output_len).unwrap()];
        let onembed_array_ref = &mut onembed_array[0];
        let dimensionality = 1;
        let ret = unsafe {
            bindings::cufftPlanMany(
                std::ptr::from_mut(plan_ref),
                dimensionality,
                std::ptr::from_mut(n_ref),
                std::ptr::from_mut(inembed_array_ref),
                1,
                c_int::try_from(input_distance).unwrap(),
                std::ptr::from_mut(onembed_array_ref),
                1,
                c_int::try_from(output_distance).unwrap(),
                bindings::cufftType_t_CUFFT_C2C,
                c_int::try_from(batch_size).unwrap(),
            )
        };
        ensure_cufft_success!(ret, "cufftPlanMany");

        let ret = unsafe { bindings::cufftSetStream(plan, stream.cu_stream().cast::<_>()) };
        ensure_cufft_success!(ret, "cufftSetStream");

        Ok(CuFFTPlan {
            plan,
            stream: Arc::clone(stream),
            required_input_len,
            required_output_len,
        })
    }

    pub fn execute_c2c(
        &self,
        input: CudaView<'_, f32>,
        mut output: CudaViewMut<'_, f32>,
        direction: c_int,
    ) -> Result<()> {
        anyhow::ensure!(
            input.len() == 2 * self.required_input_len,
            "input slice does not have required length; got {} need {}",
            input.len(),
            2 * self.required_input_len
        );
        anyhow::ensure!(
            output.len() == 2 * self.required_output_len,
            "output slice does not have required length; got {} need {}",
            output.len(),
            2 * self.required_output_len
        );

        let (input_ptr, _input_sync) = input.device_ptr(&self.stream);
        let (output_ptr, _output_sync) = output.device_ptr_mut(&self.stream);
        let ret = unsafe {
            bindings::cufftExecC2C(
                self.plan,
                input_ptr as *mut bindings::cufftComplex,
                output_ptr as *mut bindings::cufftComplex,
                direction,
            )
        };
        ensure_cufft_success!(ret, "cufftExecC2C");
        Ok(())
    }

    pub fn execute_c2c_forward(
        &self,
        input: CudaView<'_, f32>,
        output: CudaViewMut<'_, f32>,
    ) -> Result<()> {
        self.execute_c2c(input, output, bindings::CUFFT_FORWARD)
    }

    pub fn execute_c2c_inverse(
        &self,
        input: CudaView<'_, f32>,
        output: CudaViewMut<'_, f32>,
    ) -> Result<()> {
        self.execute_c2c(
            input,
            output,
            c_int::try_from(bindings::CUFFT_INVERSE).unwrap(),
        )
    }
}
