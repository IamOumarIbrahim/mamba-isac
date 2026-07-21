import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import numpy as np
import matplotlib.pyplot as plt

# Set publication quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

def generate_all_plots(results_dir: str = "results", output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating publication figures in '{output_dir}/'...")

    # 1. NMSE vs SNR Plot
    snr_file = os.path.join(results_dir, "ablation_snr.csv")
    if os.path.exists(snr_file):
        snr_vals, lmmse_nmse, trans_nmse, mamba_nmse = [], [], [], []
        with open(snr_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                snr_vals.append(float(row['SNR_dB']))
                lmmse_nmse.append(float(row['LMMSE_NMSE']))
                trans_nmse.append(float(row['Transformer_NMSE']))
                mamba_nmse.append(float(row['Mamba_NMSE']))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(snr_vals, lmmse_nmse, 'o--', label='LMMSE Baseline', color='#e74c3c')
        ax.plot(snr_vals, trans_nmse, 's-', label='Transformer Baseline', color='#3498db')
        ax.plot(snr_vals, mamba_nmse, '^-', label='Mamba-ISAC (Proposed)', color='#2ecc71')

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Comm Channel NMSE (dB)')
        ax.set_title('Communication Channel NMSE vs. Operating SNR')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "nmse_vs_snr.pdf")
        plt.savefig(fig_path, dpi=300)
        plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")

    # 2. Latency vs Sequence Length Plot
    seq_file = os.path.join(results_dir, "ablation_seq_length.csv")
    if os.path.exists(seq_file):
        seq_lens, lmmse_lat, trans_lat, mamba_lat = [], [], [], []
        with open(seq_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_lens.append(int(row['T']))
                lmmse_lat.append(float(row['LMMSE_Lat_ms']))
                trans_lat.append(float(row['Transformer_Lat_ms']))
                mamba_lat.append(float(row['Mamba_Lat_ms']))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(seq_lens, lmmse_lat, 'o--', label='LMMSE', color='#e74c3c')
        ax.plot(seq_lens, trans_lat, 's-', label='Transformer O(T^2)', color='#3498db')
        ax.plot(seq_lens, mamba_lat, '^-', label='Mamba-ISAC O(T)', color='#2ecc71')

        ax.set_xlabel('Sequence Length / Pilot Slots (T)')
        ax.set_ylabel('Inference Latency (ms)')
        ax.set_title('Inference Latency vs. Sequence Length T')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "latency_vs_seq_length.pdf")
        plt.savefig(fig_path, dpi=300)
        plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")

    # 3. Mobility Sweep Plot
    mob_file = os.path.join(results_dir, "ablation_mobility.csv")
    if os.path.exists(mob_file):
        vel_vals, lmmse_nmse, trans_nmse, mamba_nmse = [], [], [], []
        with open(mob_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vel_vals.append(float(row['velocity_kmh']))
                lmmse_nmse.append(float(row['LMMSE_NMSE']))
                trans_nmse.append(float(row['Transformer_NMSE']))
                mamba_nmse.append(float(row['Mamba_NMSE']))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(vel_vals, lmmse_nmse, 'o--', label='LMMSE Baseline', color='#e74c3c')
        ax.plot(vel_vals, trans_nmse, 's-', label='Transformer Baseline', color='#3498db')
        ax.plot(vel_vals, mamba_nmse, '^-', label='Mamba-ISAC (Proposed)', color='#2ecc71')

        ax.set_xlabel('Target Velocity (km/h)')
        ax.set_ylabel('Comm Channel NMSE (dB)')
        ax.set_title('Mobility Sweep: Channel Estimation Robustness')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "mobility_sweep.pdf")
        plt.savefig(fig_path, dpi=300)
        plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")

    # 4. Pilot Density Sweep Plot
    pil_file = os.path.join(results_dir, "ablation_pilot_density.csv")
    if os.path.exists(pil_file):
        oh_vals, lmmse_nmse, trans_nmse, mamba_nmse = [], [], [], []
        with open(pil_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oh_vals.append(float(row['pilot_overhead_pct']))
                lmmse_nmse.append(float(row['LMMSE_NMSE']))
                trans_nmse.append(float(row['Transformer_NMSE']))
                mamba_nmse.append(float(row['Mamba_NMSE']))

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(oh_vals, lmmse_nmse, 'o--', label='LMMSE Baseline', color='#e74c3c')
        ax.plot(oh_vals, trans_nmse, 's-', label='Transformer Baseline', color='#3498db')
        ax.plot(oh_vals, mamba_nmse, '^-', label='Mamba-ISAC (Proposed)', color='#2ecc71')

        ax.set_xlabel('Pilot Overhead Ratio (%)')
        ax.set_ylabel('Comm Channel NMSE (dB)')
        ax.set_title('Pilot Density Sweep: Overhead vs. Accuracy')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "pilot_density_sweep.pdf")
        plt.savefig(fig_path, dpi=300)
        plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")

    print("All figures successfully generated and saved!")

if __name__ == "__main__":
    generate_all_plots()
