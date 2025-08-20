import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- Parse iperf_cake_tcp.log (TCP) ---
def parse_iperf_log(filename):
    data = {
        'time': [],
        'stream6_bitrate': [],
        'stream8_bitrate': [],
        'stream10_bitrate': [],
        'stream12_bitrate': [],
        'sum_bitrate': [],
        'stream6_retr': [],
        'stream8_retr': [],
        'stream10_retr': [],
        'stream12_retr': [],
        'sum_retr': [],
        'stream6_cwnd': [],
        'stream8_cwnd': [],
        'stream10_cwnd': [],
        'stream12_cwnd': [],
    }
    final_stats = {}

    interval_pattern = re.compile(
        r"^\[\s*(\d+|SUM)\]\s+([\d.]+)-([\d.]+)\s+sec\s+([\d.]+)\s+(KBytes|MBytes|GBytes)\s+([\d.]+)\s+(Kbits/sec|Mbits/sec|Gbits/sec)(?:\s+(\d+)(?:\s+([\d.]+)\s+(KBytes|MBytes))?)?"
    )
    
    summary_pattern = re.compile(
        "\[\s*(SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)"
    )

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            interval_match = interval_pattern.search(line)
            if interval_match:
                stream_id = interval_match.group(1)
                interval_end = float(interval_match.group(3))
                transfer_val = float(interval_match.group(4))
                transfer_unit = interval_match.group(5)
                bitrate_val = float(interval_match.group(6))
                bitrate_unit = interval_match.group(7)
                
                # Convert all to Mbits/sec and MBytes
                if transfer_unit == 'KBytes':
                    transfer_val /= 1024
                elif transfer_unit == 'GBytes':
                    transfer_val *= 1024
                
                if bitrate_unit == 'Kbits/sec':
                    bitrate_val /= 1024
                elif bitrate_unit == 'Gbits/sec':
                    bitrate_val *= 1024
                
                retransmits = int(interval_match.group(8)) if interval_match.group(8) else 0
                
                cwnd = 0
                if interval_match.group(9) and interval_match.group(10):
                    cwnd = float(interval_match.group(9))
                    if interval_match.group(10) == 'MBytes':
                        cwnd *= 1024

                if stream_id == 'SUM':
                    data['time'].append(interval_end)
                    data['sum_bitrate'].append(bitrate_val)
                    data['sum_retr'].append(retransmits)
                elif stream_id.isdigit():
                    data[f'stream{stream_id}_bitrate'].append(bitrate_val)
                    data[f'stream{stream_id}_retr'].append(retransmits)
                    data[f'stream{stream_id}_cwnd'].append(cwnd)

            summary_match = summary_pattern.search(line)
            if summary_match:
                if summary_match.group(1) == 'SUM':
                    final_stats['total_transfer_tcp'] = float(summary_match.group(4))
                    final_stats['total_bitrate_tcp'] = float(summary_match.group(5))
                    final_stats['total_retr_tcp'] = int(summary_match.group(6))

    return data, final_stats

# --- Parse iperf_cake_udp.log (UDP) ---
def parse_iperf_log_udp(filename):
    data = {
        'time': [],
        'stream6_transfer': [], 'stream8_transfer': [], 'stream10_transfer': [], 'stream12_transfer': [],
        'sum_transfer': [],
        'stream6_bitrate': [], 'stream8_bitrate': [], 'stream10_bitrate': [], 'stream12_bitrate': [],
        'sum_bitrate': [],
        'stream6_datagrams': [], 'stream8_datagrams': [], 'stream10_datagrams': [], 'stream12_datagrams': [],
        'sum_datagrams': [],
        'stream6_lost_datagrams': [], 'stream8_lost_datagrams': [], 'stream10_lost_datagrams': [], 'stream12_lost_datagrams': [],
        'sum_lost_datagrams': [],
        'stream6_total_datagrams': [], 'stream8_total_datagrams': [], 'stream10_total_datagrams': [], 'stream12_total_datagrams': [],
        'sum_total_datagrams': [],
        'stream6_jitter': [], 'stream8_jitter': [], 'stream10_jitter': [], 'stream12_jitter': [], 'sum_jitter': []
    }
    final_stats = {}

    interval_pattern = re.compile(
        r"^\[\s*(\d+|SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+(KBytes|MBytes|GBytes)\s+([\d\.]+)\s+(Kbits/sec|Mbits/sec|Gbits/sec)\s+(\d+)\s*$"
    )
    
    summary_pattern = re.compile(
        r"^\[\s*(SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+([\d\.]+)\s+ms\s+(\d+)/(\d+)\s+\(([\d\.]+)%\)\s+receiver\s*$"
    )

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            interval_match = interval_pattern.search(line)
            if interval_match:
                stream_id = interval_match.group(1)
                interval_end = float(interval_match.group(3))
                transfer_val = float(interval_match.group(4))
                transfer_unit = interval_match.group(5)
                bitrate_val = float(interval_match.group(6))
                bitrate_unit = interval_match.group(7)
                total_datagrams = int(interval_match.group(8))
                
                # Convert all to Mbits/sec and MBytes
                if transfer_unit == 'KBytes':
                    transfer_val /= 1024
                elif transfer_unit == 'GBytes':
                    transfer_val *= 1024

                if bitrate_unit == 'Kbits/sec':
                    bitrate_val /= 1024
                elif bitrate_unit == 'Gbits/sec':
                    bitrate_val *= 1024

                if stream_id == 'SUM':
                    data['time'].append(interval_end)
                    data['sum_transfer'].append(transfer_val)
                    data['sum_bitrate'].append(bitrate_val)
                    data['sum_datagrams'].append(total_datagrams)
                    data['sum_lost_datagrams'].append(0)
                    data['sum_total_datagrams'].append(total_datagrams)
                    data['sum_jitter'].append(0.0)
                elif stream_id.isdigit():
                    data[f'stream{stream_id}_transfer'].append(transfer_val)
                    data[f'stream{stream_id}_bitrate'].append(bitrate_val)
                    data[f'stream{stream_id}_datagrams'].append(total_datagrams)
                    data[f'stream{stream_id}_lost_datagrams'].append(0)
                    data[f'stream{stream_id}_total_datagrams'].append(total_datagrams)
                    data[f'stream{stream_id}_jitter'].append(0.0)

            summary_match = summary_pattern.search(line)
            if summary_match:
                if summary_match.group(1) == 'SUM':
                    final_stats['total_transfer_udp'] = float(summary_match.group(4))
                    final_stats['total_bitrate_udp'] = float(summary_match.group(5))
                    final_stats['avg_jitter_udp'] = float(summary_match.group(6))
                    final_stats['lost_datagrams_udp'] = int(summary_match.group(7))
                    final_stats['total_datagrams_udp'] = int(summary_match.group(8))
                    final_stats['loss_percent_udp'] = float(summary_match.group(9))

    return data, final_stats

# --- Parse ping_cake.log ---
def parse_ping_log(filename):
    times = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search("time=([\d\.]+) ms", line)
            if match:
                times.append(float(match.group(1)))
    return times

# --- Parse cake_stats.txt ---
def parse_cake_stats(filename):
    intervals = []
    current_stats = {}
    
    timestamp_pattern = re.compile(r"^(\d{2}:\d{2}:\d{2})$")
    sent_dropped_pattern = re.compile(r"Sent (\d+) bytes (\d+) pkt \(dropped (\d+), overlimits (\d+) requeues (\d+)\)")
    memory_pattern = re.compile(r"memory used: (\d+)b of (\d+)(K|M|G)b")
    capacity_pattern = re.compile(r"capacity estimate: (\d+)(Kbit|Mbit|Gbit)")
    
    def convert_unit(value, unit):
        if unit == 'us':
            return value / 1000
        elif unit == 'Gbit':
            return value * 1024
        elif unit == 'Kbit':
            return value / 1024
        return value

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            timestamp_match = timestamp_pattern.search(line)

            if timestamp_match:
                if current_stats:
                    intervals.append(current_stats.copy())
                current_stats = {'time': timestamp_match.group(1)}
                i += 1
                continue

            sent_dropped_match = sent_dropped_pattern.search(line)
            if sent_dropped_match:
                current_stats['sent_bytes'] = int(sent_dropped_match.group(1))
                current_stats['sent_pkt'] = int(sent_dropped_match.group(2))
                current_stats['dropped_pkt'] = int(sent_dropped_match.group(3))
                current_stats['overlimits'] = int(sent_dropped_match.group(4))
                current_stats['requeues'] = int(sent_dropped_match.group(5))

            memory_match = memory_pattern.search(line)
            if memory_match:
                current_stats['memory_used_b'] = int(memory_match.group(1))
                memory_unit = memory_match.group(3)
                memory_multiplier = 1
                if memory_unit == 'K':
                    memory_multiplier = 1024
                elif memory_unit == 'M':
                    memory_multiplier = 1024 * 1024
                elif memory_unit == 'G':
                    memory_multiplier = 1024 * 1024 * 1024
                current_stats['memory_total_b'] = int(memory_match.group(2)) * memory_multiplier

            capacity_match = capacity_pattern.search(line)
            if capacity_match:
                current_stats['capacity_estimate_mbps'] = convert_unit(float(capacity_match.group(1)), capacity_match.group(2))

            if "Bulk  Best Effort        Video        Voice" in line:
                j = i + 1
                while j < len(lines) and lines[j].strip() and not timestamp_pattern.search(lines[j]):
                    tc_line = lines[j].strip()
                    
                    if "pk_delay" in tc_line:
                        pk_delay_match = re.search(r"pk_delay\s+(\d+)(us|ms)\s+(\d+)(us|ms)\s+(\d+)(us|ms)\s+(\d+)(us|ms)", tc_line)
                        if pk_delay_match:
                            current_stats['pk_delay_be'] = convert_unit(float(pk_delay_match.group(3)), pk_delay_match.group(4))
                            current_stats['pk_delay_video'] = convert_unit(float(pk_delay_match.group(5)), pk_delay_match.group(6))
                            current_stats['pk_delay_voice'] = convert_unit(float(pk_delay_match.group(7)), pk_delay_match.group(8))
                    
                    elif "av_delay" in tc_line:
                        av_delay_match = re.search(r"av_delay\s+(\d+)(us|ms)?\s+(\d+)(us|ms)?\s+(\d+)(us|ms)?\s+(\d+)(us|ms)?", tc_line)
                        if av_delay_match:
                            current_stats['av_delay_be'] = convert_unit(float(av_delay_match.group(3)), av_delay_match.group(4) if av_delay_match.group(4) else 'us')
                            current_stats['av_delay_video'] = convert_unit(float(av_delay_match.group(5)), av_delay_match.group(6) if av_delay_match.group(6) else 'us')
                            current_stats['av_delay_voice'] = convert_unit(float(av_delay_match.group(7)), av_delay_match.group(8) if av_delay_match.group(8) else 'us')
                    
                    elif "drops" in tc_line:
                        drops_match = re.search(r"drops\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", tc_line)
                        if drops_match:
                            current_stats['drops_be'] = int(drops_match.group(2))
                            current_stats['drops_video'] = int(drops_match.group(3))
                            current_stats['drops_voice'] = int(drops_match.group(4))
                    
                    elif "marks" in tc_line:
                        marks_match = re.search(r"marks\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", tc_line)
                        if marks_match:
                            current_stats['marks_be'] = int(marks_match.group(2))
                            current_stats['marks_video'] = int(marks_match.group(3))
                            current_stats['marks_voice'] = int(marks_match.group(4))
                    
                    elif "pkts" in tc_line:
                        pkts_match = re.search(r"pkts\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", tc_line)
                        if pkts_match:
                            current_stats['pkts_be'] = int(pkts_match.group(2))
                            current_stats['pkts_video'] = int(pkts_match.group(3))
                            current_stats['pkts_voice'] = int(pkts_match.group(4))
                    
                    j += 1
                i = j - 1
            i += 1

    if current_stats:
        intervals.append(current_stats.copy())

    return intervals

# --- Analysis functions ---
def analyze_bandwidth(total_bitrate, target_bandwidth):
    efficiency = (total_bitrate / target_bandwidth) * 100
    return efficiency

def analyze_packet_loss(sent_pkt, dropped):
    if sent_pkt > 0:
        loss_rate = (dropped / sent_pkt) * 100
    else:
        loss_rate = 0
    return loss_rate

def analyze_latency(times):
    if not times:
        return {'min': 0, 'avg': 0, 'max': 0, 'std': 0}
    min_time = np.min(times)
    avg_time = np.mean(times)
    max_time = np.max(times)
    std_dev = np.std(times)
    return {'min': min_time, 'avg': avg_time, 'max': max_time, 'std': std_dev}

def analyze_fairness(flow_type, values):
    if not values:
        return 0.0
    values = np.array(values)
    numerator = np.sum(values)**2
    denominator = len(values) * np.sum(values**2)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def analyze_fairness_wrapper(flow_type, stream_bitrates=None):
    fairness_results = {}
    if stream_bitrates and all(bitrate is not None for bitrate in stream_bitrates):
        fairness_results[f'jains_index_iperf_{flow_type}'] = analyze_fairness(flow_type, stream_bitrates)
    else:
        fairness_results[f'jains_index_iperf_{flow_type}'] = 0.0
    return fairness_results

# --- Optimized plotting functions ---
def plot_cake_analysis_tcp(data, final_stats, ping_times, out_img):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10})
    
    n_plots = 6 if ping_times else 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    fig.suptitle('PHÂN TÍCH HIỆU NĂNG CAKE TCP', fontsize=16, fontweight='bold', y=0.98)

    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Get data and ensure consistency
    min_length = min(
        len(data.get('time', [])),
        *(len(data.get(f'stream{i}_bitrate', [])) for i in [6, 8, 10, 12]),
        len(data.get('sum_bitrate', []))
    ) if data.get('time') else 0
    
    time_data = data.get('time', [])[:min_length] if min_length > 0 else []
    stream_bitrates = [data.get(f'stream{i}_bitrate', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    sum_bitrate = data.get('sum_bitrate', [])[:min_length] if min_length > 0 else []
    stream_retrs = [data.get(f'stream{i}_retr', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    sum_retr = data.get('sum_retr', [])[:min_length] if min_length > 0 else []
    stream_cwnd = [data.get(f'stream{i}_cwnd', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]

    # Plot 1: Bitrate over time
    ax = axs[0]
    if time_data and any(stream_bitrates) and sum_bitrate:
        for i, stream_bitrate in enumerate(stream_bitrates):
            if stream_bitrate and len(stream_bitrate) == len(time_data):
                ax.plot(time_data, stream_bitrate, label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
        if len(sum_bitrate) == len(time_data):
            ax.plot(time_data, sum_bitrate, 'k--', label='Tổng', linewidth=3, alpha=0.9)
        ax.set_title('Bitrate TCP theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Bitrate (Mbits/sec)')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(time_data) if time_data else 1)
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Bitrate TCP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Bitrate TCP theo thời gian', fontweight='bold')

    # Plot 2: Retransmissions over time
    ax = axs[1]
    if time_data and any(stream_retrs):
        cumulative_retrs = []
        for i, stream_retr in enumerate(stream_retrs):
            if stream_retr and len(stream_retr) == len(time_data):
                cumulative = np.cumsum(stream_retr)
                cumulative_retrs.append(cumulative)
                ax.plot(time_data, cumulative, label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
        
        if sum_retr and len(sum_retr) == len(time_data):
            cumulative_sum = np.cumsum(sum_retr)
            ax.plot(time_data, cumulative_sum, 'k--', label='Tổng', linewidth=3, alpha=0.9)
        
        ax.set_title('Gói truyền lại TCP tích lũy theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Số gói truyền lại tích lũy')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(time_data) if time_data else 1)
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu gói truyền lại TCP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Gói truyền lại TCP theo thời gian', fontweight='bold')

    # Plot 3: Congestion Window
    ax = axs[2]
    if time_data and any(stream_cwnd):
        for i, stream_c in enumerate(stream_cwnd):
            if stream_c and len(stream_c) == len(time_data):
                ax.plot(time_data, stream_c, label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
        ax.set_title('Cửa sổ tắc nghẽn (Cwnd) TCP theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Cwnd (KBytes)')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(time_data) if time_data else 1)
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Cwnd TCP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Cửa sổ tắc nghẽn (Cwnd) TCP theo thời gian', fontweight='bold')

    # Plot 4: Ping Latency (if available)
    if ping_times and len(axs) > 3:
        ax = axs[3]
        ping_x = list(range(len(ping_times)))
        ax.plot(ping_x, ping_times, 'b-', linewidth=2, alpha=0.8, label='Ping')
        ax.fill_between(ping_x, ping_times, alpha=0.3, color='blue')
        ax.set_title('Độ trễ Ping theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Số thứ tự Ping')
        ax.set_ylabel('Độ trễ (ms)')
        ax.grid(True, alpha=0.3)
        
        # Add statistical info
        if ping_times:
            avg_ping = sum(ping_times) / len(ping_times)
            max_ping = max(ping_times)
            min_ping = min(ping_times)
            ax.axhline(y=avg_ping, color='red', linestyle='--', alpha=0.7, 
                      label=f'TB: {avg_ping:.2f}ms')
            ax.axhline(y=max_ping, color='orange', linestyle=':', alpha=0.7, 
                      label=f'Max: {max_ping:.2f}ms')
            ax.axhline(y=min_ping, color='green', linestyle=':', alpha=0.7, 
                      label=f'Min: {min_ping:.2f}ms')
            ax.legend(loc='best')
    elif len(axs) > 3:
        ax = axs[3]
        ax.text(0.5, 0.5, 'Không có dữ liệu độ trễ Ping', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Độ trễ Ping theo thời gian', fontweight='bold')

    # Plot 5: Bitrate Distribution
    ax = axs[n_plots - 2]
    all_bitrates = []
    for stream_bitrate in stream_bitrates:
        if stream_bitrate:
            all_bitrates.extend(stream_bitrate)
    
    if all_bitrates:
        n_bins = min(30, len(set(all_bitrates)))
        counts, bins, patches = ax.hist(all_bitrates, bins=n_bins, alpha=0.7, color='skyblue', 
                                       edgecolor='black', linewidth=1)
        ax.set_title('Phân bố Bitrate TCP giữa các luồng', fontweight='bold', pad=15)
        ax.set_xlabel('Bitrate (Mbits/sec)')
        ax.set_ylabel('Tần suất')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistical info
        mean_bitrate = np.mean(all_bitrates)
        std_bitrate = np.std(all_bitrates)
        ax.axvline(x=mean_bitrate, color='red', linestyle='--', alpha=0.7, 
                  label=f'TB: {mean_bitrate:.2f} ± {std_bitrate:.2f} Mbps')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu phân bố Bitrate TCP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Phân bố Bitrate TCP giữa các luồng', fontweight='bold')

    # Plot 6: TCP Throughput (properly calculated)
    ax = axs[n_plots - 1]
    if time_data and any(stream_bitrates):
        throughput_data = {}
        for i, stream in enumerate(streams):
            stream_bitrate = stream_bitrates[i]
            if stream_bitrate and len(stream_bitrate) == len(time_data):
                # Calculate throughput as MBytes transferred per interval
                throughput = []
                for j, bitrate in enumerate(stream_bitrate):
                    if j > 0:
                        interval = time_data[j] - time_data[j-1]
                        # Convert Mbits/sec to MBytes: (Mbits/sec * interval_sec) / 8
                        throughput.append((bitrate * interval) / 8)
                    else:
                        throughput.append(0)
                throughput_data[stream] = np.cumsum(throughput)

        if throughput_data:
            for i, stream in enumerate(streams):
                if stream in throughput_data:
                    ax.plot(time_data, throughput_data[stream], label=labels[i], 
                           color=colors[i], linewidth=2, alpha=0.8)
            ax.set_title('Tổng dữ liệu truyền TCP theo thời gian', fontweight='bold', pad=15)
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Dữ liệu tích lũy (MBytes)')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(time_data) if time_data else 1)
        else:
            ax.text(0.5, 0.5, 'Không có dữ liệu Transfer TCP', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Transfer TCP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Tổng dữ liệu truyền TCP theo thời gian', fontweight='bold')

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    
    # Add enhanced summary text
    summary_text = f"""📊 CAKE TCP PERFORMANCE SUMMARY
• Total Bitrate: {final_stats.get('total_bitrate_tcp', 0.0):.2f} Mbits/sec | Efficiency: {final_stats.get('efficiency_tcp', 0.0):.1f}%
• Retransmissions: {final_stats.get('total_retr_tcp', 0)} packets | Fairness Index: {final_stats.get('jains_index_iperf_tcp', 0.0):.3f}
• Latency - Min: {final_stats.get('latency_tcp', {}).get('min', 0.0):.2f}ms | Avg: {final_stats.get('latency_tcp', {}).get('avg', 0.0):.2f}ms | Max: {final_stats.get('latency_tcp', {}).get('max', 0.0):.2f}ms"""
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_cake_analysis_udp(data, final_stats, ping_times, out_img):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10})
    
    n_plots = 5 if ping_times else 4
    fig, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    fig.suptitle('PHÂN TÍCH HIỆU NĂNG CAKE UDP', fontsize=16, fontweight='bold', y=0.98)

    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    min_length = min(
        len(data.get('time', [])),
        *(len(data.get(f'stream{i}_bitrate', [])) for i in [6, 8, 10, 12]),
        len(data.get('sum_bitrate', []))
    ) if data.get('time') else 0
    
    time_data = data.get('time', [])[:min_length] if min_length > 0 else []
    stream_bitrates = [data.get(f'stream{i}_bitrate', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    sum_bitrate = data.get('sum_bitrate', [])[:min_length] if min_length > 0 else []

    # Plot 1: Bitrate over time
    ax = axs[0]
    if time_data and any(stream_bitrates) and sum_bitrate:
        for i, stream_bitrate in enumerate(stream_bitrates):
            if stream_bitrate and len(stream_bitrate) == len(time_data):
                ax.plot(time_data, stream_bitrate, label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
        if len(sum_bitrate) == len(time_data):
            ax.plot(time_data, sum_bitrate, 'k--', label='Tổng', linewidth=3, alpha=0.9)
        ax.set_title('Bitrate UDP theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Bitrate (Mbits/sec)')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(time_data) if time_data else 1)
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Bitrate UDP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Bitrate UDP theo thời gian', fontweight='bold')

    # Plot 2: Packet Loss Analysis
    ax = axs[1]
    if final_stats.get('loss_percent_udp', 0) > 0:
        # Create a simple visualization of packet loss
        categories = ['Packets Sent', 'Packets Lost']
        values = [final_stats.get('total_datagrams_udp', 0) - final_stats.get('lost_datagrams_udp', 0),
                 final_stats.get('lost_datagrams_udp', 0)]
        colors_loss = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors_loss, alpha=0.7, edgecolor='black')
        ax.set_title('Phân tích mất gói UDP', fontweight='bold', pad=15)
        ax.set_ylabel('Số gói')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total = sum(values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                   f'{height:,}\n({percentage:.2f}%)', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu mất gói UDP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Phân tích mất gói UDP', fontweight='bold')

    # Plot 3: Ping Latency (if available)
    if ping_times and len(axs) > 2:
        ax = axs[2]
        ping_x = list(range(len(ping_times)))
        ax.plot(ping_x, ping_times, 'b-', linewidth=2, alpha=0.8, label='Ping')
        ax.fill_between(ping_x, ping_times, alpha=0.3, color='blue')
        ax.set_title('Độ trễ Ping theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Số thứ tự Ping')
        ax.set_ylabel('Độ trễ (ms)')
        ax.grid(True, alpha=0.3)
        
        if ping_times:
            avg_ping = sum(ping_times) / len(ping_times)
            ax.axhline(y=avg_ping, color='red', linestyle='--', alpha=0.7, 
                      label=f'TB: {avg_ping:.2f}ms')
            ax.legend(loc='best')
    elif len(axs) > 2:
        ax = axs[2]
        ax.text(0.5, 0.5, 'Không có dữ liệu độ trễ Ping', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Độ trễ Ping theo thời gian', fontweight='bold')

    # Plot 4: UDP Throughput
    ax = axs[n_plots - 1]
    if time_data and any(stream_bitrates):
        throughput_data = {}
        for i, stream in enumerate(streams):
            stream_bitrate = stream_bitrates[i]
            if stream_bitrate and len(stream_bitrate) == len(time_data):
                throughput = []
                for j, bitrate in enumerate(stream_bitrate):
                    if j > 0:
                        interval = time_data[j] - time_data[j-1]
                        throughput.append((bitrate * interval) / 8)
                    else:
                        throughput.append(0)
                throughput_data[stream] = np.cumsum(throughput)

        if throughput_data:
            for i, stream in enumerate(streams):
                if stream in throughput_data:
                    ax.plot(time_data, throughput_data[stream], label=labels[i], 
                           color=colors[i], linewidth=2, alpha=0.8)
            ax.set_title('Tổng dữ liệu truyền UDP theo thời gian', fontweight='bold', pad=15)
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Dữ liệu tích lũy (MBytes)')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(time_data) if time_data else 1)
        else:
            ax.text(0.5, 0.5, 'Không có dữ liệu Transfer UDP', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Transfer UDP', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Tổng dữ liệu truyền UDP theo thời gian', fontweight='bold')

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])

    # Add enhanced summary text
    summary_text = f"""📊 CAKE UDP PERFORMANCE SUMMARY
• Total Bitrate: {final_stats.get('total_bitrate_udp', 0.0):.2f} Mbits/sec | Efficiency: {final_stats.get('efficiency_udp', 0.0):.1f}%
• Jitter: {final_stats.get('avg_jitter_udp', 0.0):.3f}ms | Packet Loss: {final_stats.get('loss_percent_udp', 0.0):.3f}% ({final_stats.get('lost_datagrams_udp', 0)}/{final_stats.get('total_datagrams_udp', 0)})
• Fairness Index: {final_stats.get('jains_index_iperf_udp', 0.0):.3f} | Transfer: {final_stats.get('total_transfer_udp', 0.0):.1f} MBytes"""
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
    
    plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_cake_stats_analysis(intervals, out_img):
    if not intervals:
        print("Không có dữ liệu thống kê CAKE để vẽ biểu đồ.")
        return

    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10})

    # Extract data
    times = list(range(len(intervals)))
    time_labels = [interval.get('time', f'T{i}') for i, interval in enumerate(intervals)]
    dropped_pkts = [interval.get('dropped_pkt', 0) for interval in intervals]
    overlimits = [interval.get('overlimits', 0) for interval in intervals]
    requeues = [interval.get('requeues', 0) for interval in intervals]
    
    pk_delay_be = [interval.get('pk_delay_be', 0) for interval in intervals]
    av_delay_be = [interval.get('av_delay_be', 0) for interval in intervals]
    drops_be = [interval.get('drops_be', 0) for interval in intervals]
    marks_be = [interval.get('marks_be', 0) for interval in intervals]

    num_plots = 3
    if any(pk_delay_be) or any(av_delay_be):
        num_plots += 1
    if any(drops_be):
        num_plots += 1

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 4 * num_plots))
    if num_plots == 1:
        axs = [axs]
    fig.suptitle('PHÂN TÍCH THỐNG KÊ CAKE', fontsize=16, fontweight='bold', y=0.98)

    current_plot_idx = 0

    # Plot 1: Dropped Packets Over Time
    ax = axs[current_plot_idx]
    if times and dropped_pkts:
        bars = ax.bar(times, dropped_pkts, color='red', alpha=0.7, edgecolor='black')
        ax.set_title('Gói bị loại theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số gói bị loại')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(times)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(dropped_pkts)*0.01,
                       f'{int(height)}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu gói bị loại', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Gói bị loại theo thời gian', fontweight='bold')
    current_plot_idx += 1

    # Plot 2: Overlimits Over Time
    ax = axs[current_plot_idx]
    if times and overlimits:
        bars = ax.bar(times, overlimits, color='orange', alpha=0.7, edgecolor='black')
        ax.set_title('Overlimits theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số lần overlimit')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(times)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(overlimits)*0.01,
                       f'{int(height)}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Overlimits', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Overlimits theo thời gian', fontweight='bold')
    current_plot_idx += 1

    # Plot 3: Requeues Over Time
    ax = axs[current_plot_idx]
    if times and requeues:
        bars = ax.bar(times, requeues, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title('Requeues theo thời gian', fontweight='bold', pad=15)
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số lần requeue')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(times)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(requeues)*0.01,
                       f'{int(height)}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu Requeues', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Requeues theo thời gian', fontweight='bold')
    current_plot_idx += 1

    # Plot 4: Delays (if available)
    if (any(pk_delay_be) or any(av_delay_be)) and current_plot_idx < len(axs):
        ax = axs[current_plot_idx]
        if times and (pk_delay_be or av_delay_be):
            if pk_delay_be:
                ax.plot(times, pk_delay_be, 'o-', label='Peak Delay (Best Effort)', 
                       color='red', linewidth=2, markersize=6)
            if av_delay_be:
                ax.plot(times, av_delay_be, 's-', label='Average Delay (Best Effort)', 
                       color='blue', linewidth=2, markersize=6)
            
            ax.set_title('Độ trễ theo thời gian (Best Effort)', fontweight='bold', pad=15)
            ax.set_xlabel('Thời gian')
            ax.set_ylabel('Độ trễ (ms)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(times)
            ax.set_xticklabels(time_labels, rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Không có dữ liệu độ trễ', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title('Độ trễ theo thời gian', fontweight='bold')
        current_plot_idx += 1

    # Plot 5: Drops per Traffic Class (if available)
    if any(drops_be) and current_plot_idx < len(axs):
        ax = axs[current_plot_idx]
        if times and drops_be:
            bars = ax.bar(times, drops_be, color='purple', alpha=0.7, edgecolor='black')
            ax.set_title('Gói bị loại (Best Effort) theo thời gian', fontweight='bold', pad=15)
            ax.set_xlabel('Thời gian')
            ax.set_ylabel('Số gói bị loại')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(times)
            ax.set_xticklabels(time_labels, rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(drops_be)*0.01,
                           f'{int(height)}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Không có dữ liệu gói bị loại', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title('Gói bị loại theo lớp lưu lượng', fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# --- Export báo cáo CSV ---
def export_cake_report(final_stats, output_csv):
    report_data = {
        'Metric': [],
        'Value': [],
        'Unit': []
    }

    # TCP Metrics
    report_data['Metric'].append('TCP Total Bitrate')
    report_data['Value'].append(f"{final_stats.get('total_bitrate_tcp', 0.0):.2f}")
    report_data['Unit'].append('Mbits/sec')

    report_data['Metric'].append('TCP Efficiency')
    report_data['Value'].append(f"{final_stats.get('efficiency_tcp', 0.0):.2f}")
    report_data['Unit'].append('%')

    report_data['Metric'].append('TCP Total Retransmissions')
    report_data['Value'].append(f"{final_stats.get('total_retr_tcp', 0)}")
    report_data['Unit'].append('packets')

    report_data['Metric'].append('TCP Jain\'s Fairness Index')
    report_data['Value'].append(f"{final_stats.get('jains_index_iperf_tcp', 0.0):.3f}")
    report_data['Unit'].append('-')

    # UDP Metrics
    report_data['Metric'].append('UDP Total Bitrate')
    report_data['Value'].append(f"{final_stats.get('total_bitrate_udp', 0.0):.2f}")
    report_data['Unit'].append('Mbits/sec')

    report_data['Metric'].append('UDP Efficiency')
    report_data['Value'].append(f"{final_stats.get('efficiency_udp', 0.0):.2f}")
    report_data['Unit'].append('%')

    report_data['Metric'].append('UDP Average Jitter')
    report_data['Value'].append(f"{final_stats.get('avg_jitter_udp', 0.0):.2f}")
    report_data['Unit'].append('ms')

    report_data['Metric'].append('UDP Lost Datagrams')
    report_data['Value'].append(f"{final_stats.get('lost_datagrams_udp', 0)}")
    report_data['Unit'].append('packets')

    report_data['Metric'].append('UDP Total Datagrams')
    report_data['Value'].append(f"{final_stats.get('total_datagrams_udp', 0)}")
    report_data['Unit'].append('packets')

    report_data['Metric'].append('UDP Packet Loss Rate')
    report_data['Value'].append(f"{final_stats.get('loss_percent_udp', 0.0):.2f}")
    report_data['Unit'].append('%')

    report_data['Metric'].append('UDP Jain\'s Fairness Index')
    report_data['Value'].append(f"{final_stats.get('jains_index_iperf_udp', 0.0):.3f}")
    report_data['Unit'].append('-')

    # Ping Latency Metrics
    report_data['Metric'].append('Ping Min Latency')
    report_data['Value'].append(f"{final_stats.get('latency_tcp', {}).get('min', 0.0):.2f}")
    report_data['Unit'].append('ms')

    report_data['Metric'].append('Ping Avg Latency')
    report_data['Value'].append(f"{final_stats.get('latency_tcp', {}).get('avg', 0.0):.2f}")
    report_data['Unit'].append('ms')

    report_data['Metric'].append('Ping Max Latency')
    report_data['Value'].append(f"{final_stats.get('latency_tcp', {}).get('max', 0.0):.2f}")
    report_data['Unit'].append('ms')

    report_data['Metric'].append('Ping Std Dev Latency')
    report_data['Value'].append(f"{final_stats.get('latency_tcp', {}).get('std', 0.0):.2f}")
    report_data['Unit'].append('ms')

    df = pd.DataFrame(report_data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Báo cáo đã được lưu vào: {output_csv}")

# --- Hàm chính ---
if __name__ == '__main__':
    # Đường dẫn file
    iperf_file_udp = r'cake/iperf_cake_udp.log'
    iperf_file_tcp = r'cake/iperf_cake_tcp.log'
    ping_file = r'cake/ping_cake.log'
    cake_stats_file = r'cake/cake_stats.txt'
    target_bandwidth = 100  # Mbps

    print("="*80)
    print("PHÂN TÍCH HIỆU NĂNG THUẬT TOÁN CAKE (OPTIMIZED)")
    print("="*80)

    print("\nPhân tích file iperf_cake_udp.log (UDP)...")
    data_udp, final_stats_udp = parse_iperf_log_udp(iperf_file_udp)

    print("\nPhân tích file iperf_cake_tcp.log (TCP)...")
    data_tcp, final_stats_tcp = parse_iperf_log(iperf_file_tcp)

    print("\nPhân tích file ping_cake.log ...")
    ping_times = parse_ping_log(ping_file)

    print("\nPhân tích file cake_stats.txt ...")
    cake_stats_intervals = parse_cake_stats(cake_stats_file)

    # Phân tích TCP
    print("\n--- Kết quả phân tích TCP ---")
    if 'total_bitrate_tcp' in final_stats_tcp:
        efficiency_tcp = analyze_bandwidth(final_stats_tcp['total_bitrate_tcp'], target_bandwidth)
        final_stats_tcp['efficiency_tcp'] = efficiency_tcp
        print(f"Tổng băng thông TCP: {final_stats_tcp['total_bitrate_tcp']:.2f} Mbits/sec (Hiệu suất: {efficiency_tcp:.2f}%)")
        print(f"Tổng gói truyền lại TCP: {final_stats_tcp.get('total_retr_tcp', 0)}")
    
    all_tcp_stream_bitrates = []
    for stream_id in [6, 8, 10, 12]:
        all_tcp_stream_bitrates.extend(data_tcp.get(f'stream{stream_id}_bitrate', []))

    tcp_fairness = analyze_fairness_wrapper('tcp', stream_bitrates=all_tcp_stream_bitrates)
    final_stats_tcp.update(tcp_fairness)
    print(f"Độ công bằng (Jain\'s Index) TCP: {final_stats_tcp.get('jains_index_iperf_tcp', 0.0):.3f}")

    tcp_latency_stats = analyze_latency(ping_times)
    final_stats_tcp['latency_tcp'] = tcp_latency_stats
    print(f"Độ trễ Ping (Min/Avg/Max/Std): {tcp_latency_stats['min']:.2f}/{tcp_latency_stats['avg']:.2f}/{tcp_latency_stats['max']:.2f}/{tcp_latency_stats['std']:.2f} ms")

    # Phân tích UDP
    print("\n--- Kết quả phân tích UDP ---")
    if 'total_bitrate_udp' in final_stats_udp:
        efficiency_udp = analyze_bandwidth(final_stats_udp['total_bitrate_udp'], target_bandwidth)
        final_stats_udp['efficiency_udp'] = efficiency_udp
        print(f"Tổng băng thông UDP: {final_stats_udp['total_bitrate_udp']:.2f} Mbits/sec (Hiệu suất: {efficiency_udp:.2f}%)")
        print(f"Jitter trung bình UDP: {final_stats_udp['avg_jitter_udp']:.2f} ms")
        print(f"Tổng gói gửi UDP: {final_stats_udp['total_datagrams_udp']}, Gói bị mất: {final_stats_udp['lost_datagrams_udp']}, Tỷ lệ mất gói: {final_stats_udp['loss_percent_udp']:.2f}%")

    all_udp_stream_bitrates = []
    for stream_id in [6, 8, 10, 12]:
        all_udp_stream_bitrates.extend(data_udp.get(f'stream{stream_id}_bitrate', []))
    udp_fairness = analyze_fairness_wrapper('udp', stream_bitrates=all_udp_stream_bitrates)
    final_stats_udp.update(udp_fairness)
    print(f"Độ công bằng (Jain\'s Index) UDP: {final_stats_udp.get('jains_index_iperf_udp', 0.0):.3f}")

    udp_latency_stats = analyze_latency(ping_times)
    final_stats_udp['latency_udp'] = udp_latency_stats
    print(f"Độ trễ Ping (Min/Avg/Max/Std): {udp_latency_stats['min']:.2f}/{udp_latency_stats['avg']:.2f}/{udp_latency_stats['max']:.2f}/{udp_latency_stats['std']:.2f} ms")

    # Phân tích CAKE Stats
    print("\n--- Kết quả phân tích CAKE Stats ---")
    if cake_stats_intervals:
        total_dropped_from_stats = sum(interval.get('dropped_pkt', 0) for interval in cake_stats_intervals)
        total_sent_from_stats = sum(interval.get('sent_pkt', 0) for interval in cake_stats_intervals)
        loss_rate_stats = analyze_packet_loss(total_sent_from_stats, total_dropped_from_stats)
        print(f"Tổng gói Sent (từ cake_stats): {total_sent_from_stats}, Tổng gói bị loại (từ cake_stats): {total_dropped_from_stats}, Tỷ lệ mất gói: {loss_rate_stats:.2f}%")

        avg_pk_delay_be = np.mean([interval.get('pk_delay_be', 0) for interval in cake_stats_intervals if interval.get('pk_delay_be') is not None])
        avg_av_delay_be = np.mean([interval.get('av_delay_be', 0) for interval in cake_stats_intervals if interval.get('av_delay_be') is not None])
        print(f"Avg Peak Delay (Best Effort): {avg_pk_delay_be:.2f} ms")
        print(f"Avg Average Delay (Best Effort): {avg_av_delay_be:.2f} ms")
    else:
        print("Không có dữ liệu thống kê CAKE để phân tích.")

    # Vẽ biểu đồ
    print("\nVẽ biểu đồ tối ưu hóa...")
    plot_cake_analysis_tcp(data_tcp, final_stats_tcp, ping_times, 'cake_performance_analysis_tcp_optimized.png')
    plot_cake_analysis_udp(data_udp, final_stats_udp, ping_times, 'cake_performance_analysis_udp_optimized.png')
    plot_cake_stats_analysis(cake_stats_intervals, 'cake_stats_analysis_optimized.png')

    # Xuất báo cáo CSV
    print("\nXuất báo cáo CSV...")
    combined_final_stats = {**final_stats_tcp, **final_stats_udp}
    export_cake_report(combined_final_stats, 'cake_performance_report_optimized.csv')

    print("\nQuá trình phân tích hiệu năng CAKE (tối ưu hóa) hoàn tất.") 