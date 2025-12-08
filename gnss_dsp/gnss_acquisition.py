import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sigmf

import gnss_dsp
from gnss_dsp.plots.acquisition import plot_caf, plot_snrs, plot_dopplers_modulo_1khz


def prns_list(s):
    def expand_range(s):
        if "-" in s:
            a, b = s.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(s)]

    return sorted([n for r in s.split(",") for n in expand_range(r)])


def parse_args(args=None, simulation=False):
    parser = argparse.ArgumentParser()
    if simulation:
        parser.add_argument(
            "--signal-prn",
            type=int,
            default=1,
            help="GPS PRN to use for simulated signal [default=%(default)r]",
        )
        parser.add_argument(
            "--doppler",
            type=float,
            default=0.0,
            help="Doppler to use for simulated signal (Hz) [default=%(default)r]",
        )
        parser.add_argument(
            "--cn0",
            type=float,
            default=30.0,
            help="CN0 to use for simulated signal (dB·Hz) [default=%(default)r]",
        )
        parser.add_argument(
            "--signal-time-offset",
            type=float,
            default=0.0,
            help="Time offset to use for start of symbol of simulated signal (s) [default=%(default)r]",
        )
        parser.add_argument(
            "--samp_rate",
            type=int,
            default=8_000_000,
            help="Sample rate to use for simulation (sps) [default=%(default)r]",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=0.3,
            help="Simulated signal duration (s) [default=%(default)r]",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            help="Random seed for reproducible results [default is random seed generation]",
        )
    else:
        parser.add_argument("file", type=pathlib.Path, help="SigMF input file")
        parser.add_argument(
            "--remove-half-lsb-bias",
            action="store_true",
            help="Remove 1/2 LSB bias in the input data",
        )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Read samples from the input file using this time offset (s) [default=%(default)r]",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        help="Output JSON file [default is no JSON output]",
    )
    parser.add_argument(
        "--plots-dir",
        type=pathlib.Path,
        help="Output directory for plots [default is no plots]",
    )
    parser.add_argument(
        "--caf-doppler-span",
        type=float,
        default=1000.0,
        help="Doppler span to plot in CAF (Hz) [default=%(default)r]",
    )
    parser.add_argument(
        "--caf-time-span",
        type=float,
        default=25.0,
        help="Time span to plot in CAF (usec) [default=%(default)r]",
    )
    parser.add_argument(
        "--caf-time-center",
        type=float,
        help="Use a fixed time for the center of the CAF plot (s) [default is time of CAF peak]",
    )
    parser.add_argument(
        "--prns",
        type=prns_list,
        default=prns_list("1-32"),
        help="PRNs to acquire [format is 1-7,9-10,13-25, default=1-32]",
    )
    parser.add_argument(
        "--coherent-integration",
        type=float,
        default=0.02,
        help="Coherent integration time (s) [default=%(default)r]",
    )
    parser.add_argument(
        "--non-coherent-integrations",
        type=int,
        default=14,
        help="Number of non-coherent integrations [default=%(default)r]",
    )
    parser.add_argument(
        "--min-doppler",
        type=float,
        default=-10e3,
        help="Minimum Doppler (Hz) [default=%(default)r]",
    )
    parser.add_argument(
        "--max-doppler",
        type=float,
        default=10e3,
        help="Maximum Doppler (Hz) [default=%(default)r]",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        help="Limit delay of the CAF peak search to values greater than this (s) [default is no limit]",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        help="Limit delay of the CAF peak search to values smaller than this (s) [default is no limit]",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        help="Minimum SNR for CAF plot scale [default is auto-scale]",
    )
    parser.add_argument(
        "--max-snr",
        type=float,
        help="Maximum SNR for CAF plot scale [default is auto-scale]",
    )
    parser.add_argument(
        "--doppler-oversampling",
        type=int,
        default=2,
        help="Doppler oversampling factor [default=%(default)r]",
    )
    parser.add_argument(
        "--doppler-block-size",
        type=int,
        default=8,
        help="CUDA Doppler block size [default=%(default)r]",
    )
    parser.add_argument(
        "--prns-per-block",
        type=int,
        default=4,
        help="PRNs per execution block [default=%(default)r]",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device ordinal [default=%(default)r]",
    )
    return parser.parse_args()


def remove_half_lsb_bias(dataset, samples):
    dtype = dataset.get_global_field(sigmf.sigmffile.SigMFFile.DATATYPE_KEY)
    if dtype == "ci8":
        samples += (1 + 1j) * 2**-8
    elif dtype.startswith("ci16"):
        samples += (1 + 1j) * 2**-16
    elif dtype.startswith("ci32"):
        samples += (1 + 1j) * 2**-32
    elif dtype == "ri8":
        samples += 2**-8
    elif dtype.startswith("ri16"):
        samples += 2**-16
    elif dtype.startswith("ri32"):
        samples += 2**-32
    else:
        raise ValueError(f"unknown datatype {dtype}")


def generate_simulated_signal(args):
    code = gnss_dsp.gps_l1_ca_code(args.signal_prn)
    carrier_frequency = 1575.42e6
    doppler_factor = 1 + args.doppler / carrier_frequency
    code_rate = 1.023e6 * doppler_factor
    symbol_rate = 50 * doppler_factor
    time_offset = (-args.signal_time_offset) % 20e-3
    rng = np.random.default_rng(args.random_seed)
    symbols = rng.integers(2, size=int((args.duration + time_offset) * symbol_rate + 1))
    num_samples = int(args.duration * args.samp_rate)
    modulated_code = (
        code[
            np.int32(
                (np.arange(num_samples) / args.samp_rate + time_offset) * code_rate
            )
            % code.size
        ]
        ^ symbols[
            np.int32(
                (np.arange(num_samples) / args.samp_rate + time_offset) * symbol_rate
            )
        ]
    )
    modulated_code = 1 - 2 * modulated_code.astype("float32")
    amplitude = np.sqrt(10 ** (args.cn0 / 10) * 2 / args.samp_rate)
    x = (
        amplitude
        * modulated_code
        * np.exp(
            1j * 2 * np.pi * np.arange(num_samples) * args.doppler / args.samp_rate
        )
        + rng.normal(size=num_samples)
        + 1j * rng.normal(size=num_samples)
    )
    scale = 0.01
    x = (scale * x).astype("complex64")
    return x


def extract_metadata(
    caf, acquisition, sample_offset, *, min_delay=None, max_delay=None
):
    start = 0
    end = caf.shape[1]
    if min_delay is not None:
        if max_delay is not None and max_delay < min_delay:
            raise ValueError(
                f"max-delay ({max_delay}) must be greater or equal than min-delay ({min_delay})"
            )
        start = round(acquisition.sample_rate() * min_delay)
    if max_delay is not None:
        end = round(acquisition.sample_rate() * max_delay)
        # ensure that at least we are sampling one time "column" in caf_region
        end = max(end, start + 1)
    caf_region = caf[:, start:end]
    (a, b) = np.unravel_index(np.argmax(caf_region), caf_region.shape)
    b += start
    doppler = acquisition.doppler_axis()[a]
    time = (b + sample_offset) / acquisition.sample_rate()
    caf_mean = np.mean(caf)
    caf_std = np.std(caf)
    snr = (caf[a, b] - caf_mean) / caf_std
    return {
        "max_doppler_bin": a.item(),
        "max_time_bin": b.item(),
        "doppler": doppler.item(),
        "time": time.item(),
        "caf_peak": caf[a, b].item(),
        "caf_mean": caf_mean.item(),
        "caf_std": caf_std.item(),
        "snr": snr.item(),
        "sample_offset": sample_offset,
    }


def compute_acquisition(signal, samp_rate, sample_offset, args):
    coherent_integration_ms = args.coherent_integration * 1000
    if abs(coherent_integration_ms - round(coherent_integration_ms)) > 1e-6:
        raise ValueError(
            f"coherent integration {args.coherent_integration} is not an integer number of milliseconds"
        )
    coherent_integration_ms = round(coherent_integration_ms)
    acquisition = gnss_dsp.CuFFTAcquisition(
        int(samp_rate),
        coherent_integration_ms=coherent_integration_ms,
        num_noncoherent_integrations=args.non_coherent_integrations,
        min_doppler=args.min_doppler,
        max_doppler=args.max_doppler,
        doppler_oversampling=args.doppler_oversampling,
        doppler_block_size=args.doppler_block_size,
        cuda_device_ordinal=args.cuda_device,
    )
    acquisition.set_signal(signal)
    metadata = {}
    for j in range(0, len(args.prns), args.prns_per_block):
        prn_block = args.prns[j : j + args.prns_per_block]
        results = acquisition.acquire(prn_block)
        for caf, prn in zip(results, prn_block):
            meta = extract_metadata(
                caf,
                acquisition,
                sample_offset,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
            )
            metadata[prn] = meta
            if args.plots_dir is not None:
                fig = plot_caf(
                    caf,
                    prn,
                    meta,
                    acquisition,
                    doppler_span=args.caf_doppler_span,
                    time_span=1e-6 * args.caf_time_span,
                    time_center=args.caf_time_center,
                    min_snr=args.min_snr,
                    max_snr=args.max_snr,
                )
                fig.savefig(args.plots_dir / f"CAF_G{prn:02}.png")
                plt.close(fig)
        # we need to delete caf, since otherwise it holds a reference
        # to results that keeps it alive
        del caf
        del results
    return metadata


def main(args=None, simulation=False):
    args = parse_args(args, simulation)

    if simulation:
        signal = generate_simulated_signal(args)
        samp_rate = args.samp_rate
    else:
        dataset = sigmf.sigmffile.fromfile(args.file)
        samp_rate = dataset.get_global_field(sigmf.sigmffile.SigMFFile.SAMPLE_RATE_KEY)
        signal = dataset.read_samples()
        if args.remove_half_lsb_bias:
            remove_half_lsb_bias(dataset, signal)

    sample_offset = round(samp_rate * args.time_offset)
    signal = signal[sample_offset:]

    if args.json_output is not None:
        args.json_output.parent.mkdir(exist_ok=True, parents=True)
    if args.plots_dir is not None:
        args.plots_dir.mkdir(exist_ok=True, parents=True)
    metadata = compute_acquisition(signal, samp_rate, sample_offset, args)

    if args.plots_dir is not None:
        fig = plot_snrs(metadata)
        fig.savefig(args.plots_dir / "SNR.png")
        plt.close(fig)
        fig = plot_dopplers_modulo_1khz(metadata)
        fig.savefig(args.plots_dir / "doppler_modulo_1kHz.png")
        plt.close(fig)

    if args.json_output is not None:
        json_results = {
            "acquisition_results": [
                dict(prn=f"G{prn:02d}", **metadata[prn])
                for prn in sorted(metadata.keys())
            ]
        }
        with open(args.json_output, "w") as f:
            json.dump(json_results, f, indent=4)


def main_simulation(args=None):
    main(simulation=True)
