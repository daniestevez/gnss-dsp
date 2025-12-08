import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def plot_caf(
    caf,
    prn,
    meta,
    acquisition,
    *,
    doppler_span=1000.0,
    time_span=25e-6,
    time_center=None,
    min_snr=None,
    max_snr=None,
):
    (a, b) = (meta["max_doppler_bin"], meta["max_time_bin"])
    caf_mean = meta["caf_mean"]
    caf_std = meta["caf_std"]
    fig = plt.figure()
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(4, 1), wspace=0, hspace=0
    )
    ax = fig.add_subplot(gs[0, 0])
    doppler_axis = acquisition.doppler_axis()
    doppler_bin = doppler_axis[1] - doppler_axis[0]
    a_span = round(0.5 * doppler_span / doppler_bin)
    doppler_sel = slice(max(0, a - a_span), a + a_span + 1)
    doppler_axis = doppler_axis[doppler_sel]
    b_span = round(0.5 * time_span * acquisition.sample_rate())
    b_center = b
    if time_center is not None:
        b_center = round(time_center * acquisition.sample_rate())
    time_sel = slice(max(0, b_center - b_span), b_center + b_span + 1)
    taxis = (np.arange(caf.shape[1]) + meta["sample_offset"])[
        time_sel
    ] / acquisition.sample_rate()
    ax.imshow(
        (caf[doppler_sel, time_sel][::-1] - caf_mean) / caf_std,
        aspect="auto",
        interpolation="none",
        extent=[taxis[0], taxis[-1], doppler_axis[0], doppler_axis[-1]],
        vmin=min_snr,
        vmax=max_snr,
    )
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Doppler (Hz)")
    ax0 = ax
    ax = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax.get_yaxis().set_visible(False)
    ax.plot((caf[doppler_sel, b] - caf_mean) / caf_std, doppler_axis)
    if min_snr is not None:
        ax.set_xlim(left=min_snr)
    if max_snr is not None:
        ax.set_xlim(right=max_snr)
    ax = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax.plot(taxis, (caf[a, time_sel] - caf_mean) / caf_std)
    if min_snr is not None:
        ax.set_ylim(bottom=min_snr)
    if max_snr is not None:
        ax.set_ylim(top=max_snr)
    ax.set_xlabel("Delay (s)")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.6f"))
    snr = meta["snr"]
    fig.suptitle(
        f"G{prn:02} normalized CAF\nmean = {10 * np.log10(caf_mean):.1f} dB, "
        f"std = {10 * np.log10(caf_std):.1f} dB, "
        f"SNR = {10 * np.log10(snr):.1f} dB"
    )
    return fig


def plot_snrs(metadata):
    prns = sorted(metadata.keys())
    fig, ax = plt.subplots()
    xaxis = np.arange(len(prns))
    ax.bar(xaxis, 10 * np.log10([metadata[p]["snr"] for p in prns]))
    ax.set_xticks(xaxis, [f"G{p:02d}" for p in prns], rotation=-90)
    ax.grid(axis="y")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Acquisition SNRs")
    return fig


def plot_dopplers_modulo_1khz(metadata):
    prns = sorted(metadata.keys())
    fig, ax = plt.subplots()
    xaxis = np.arange(len(prns))
    ax.plot(xaxis, np.array([metadata[p]["doppler"] for p in prns]) % 1e3, ".")
    ax.set_xticks(xaxis, [f"G{p:02d}" for p in prns], rotation=-90)
    ax.set_ylim(-50, 1050)
    ax.grid()
    ax.set_ylabel("Doppler modulo 1 kHz (Hz)")
    ax.set_title("Acquisition Doppler modulo 1 kHz")
    return fig
