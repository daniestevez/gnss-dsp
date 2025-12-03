use num_complex::Complex32;

/// Modulates a BPSK code at a given chip rate.
///
/// This function returns an iterator that corresponds to the samples obtained
/// from modulating a given BPSK code at a certain chip rate and sample
/// rate. The signal polarity convention is that a chip value of 1 corresponds
/// to a bipolar output of -1.0, and a chip value of 0 corresponds to a bipolar
/// output of 1.0. The modulation is discrete time sampling, without taking
/// aliasing into account. The modulation begins aligned with the first chip,
/// and the iterator never terminates.
pub fn bpsk_modulate(code: &[u8], chip_rate: f64, samp_rate: f64) -> impl Iterator<Item = f32> {
    let chips_per_sample = chip_rate / samp_rate;
    let mut code_phase = 0.0;
    std::iter::repeat_with(move || {
        let chip = code[code_phase as usize];
        assert!((chip == 1) || (chip == 0));
        let x = [1.0, -1.0][usize::from(chip)];
        code_phase += chips_per_sample;
        if code_phase as usize >= code.len() {
            code_phase -= code.len() as f64;
        }
        x
    })
}

/// Generates a complex local oscillator at a given frequency.
///
/// This function returns an iterator that corresponds to the samples for a
/// local oscillator (complex exponential) at a given frequency.
pub fn local_oscillator(frequency: f64, samp_rate: f64) -> impl Iterator<Item = Complex32> {
    let rad_per_sample = 2.0 * std::f64::consts::PI * frequency / samp_rate;
    let mut phase = 0.0;
    std::iter::repeat_with(move || {
        let phi = phase as f32;
        let z = Complex32::new(phi.cos(), phi.sin());
        phase += rad_per_sample;
        if phase >= std::f64::consts::PI {
            phase -= 2.0 * std::f64::consts::PI;
        } else if phase < -std::f64::consts::PI {
            phase += 2.0 * std::f64::consts::PI;
        }
        z
    })
}

/// Frequency shifts a signal.
///
/// This function performs a frequency shift to a signal. The signal is given as
/// an iterator, and the function also outputs an iterator.
pub fn frequency_shift<I: Iterator<Item = Complex32>>(
    samples: I,
    frequency: f64,
    samp_rate: f64,
) -> impl Iterator<Item = Complex32> {
    samples
        .zip(local_oscillator(frequency, samp_rate))
        .map(|(z, w)| z * w)
}

/// Converts a real signal to complex.
///
/// This function converts a real signal, given as an iterator, into a complex
/// signal, also given as an iterator, by putting the signal in the real part.
pub fn real_to_complex<I: Iterator<Item = f32>>(samples: I) -> impl Iterator<Item = Complex32> {
    samples.map(|x| Complex32::new(x, 0.0))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bspsk_modulate_simple() {
        let result = bpsk_modulate(&[0, 1], 0.5, 1.0).take(8).collect::<Vec<_>>();
        let expected = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        assert_eq!(result, expected);
    }
}
