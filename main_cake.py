
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

    # Regex: [ ID] Interval Transfer Bitrate Retr Cwnd
    # Example: [  6]   0.00-1.00   sec   892 KBytes  7.31 Mbits/sec    0   19.8 KBytes
    # Group 1: Stream ID or SUM
    # Group 2: Interval Start (unused)
    # Group 3: Interval End (time)
    # Group 4: Transfer Value
    # Group 5: Transfer Unit (KBytes, MBytes, GBytes)
    # Group 6: Bitrate Value
    # Group 7: Bitrate Unit (Kbits/sec, Mbits/sec, Gbits/sec)
    # Group 8: Retransmits
    # Group 9: Cwnd Value
    # Group 10: Cwnd Unit (KBytes, MBytes)
    interval_pattern = re.compile(
        r"^\[\s*(\d+|SUM)\]\s+([\d.]+)-([\d.]+)\s+sec\s+([\d.]+)\s+(KBytes|MBytes|GBytes)\s+([\d.]+)\s+(Kbits/sec|Mbits/sec|Gbits/sec)(?:\s+(\d+)(?:\s+([\d.]+)\s+(KBytes|MBytes))?)?"
    )
    
    # Regex for summary line: [SUM] 0.00-10.00 sec 23.9 MBytes 20.0 Mbits/sec 0
    # Group 1: SUM
    # Group 2: Start time
    # Group 3: End time
    # Group 4: Total Transfer
    # Group 5: Total Bitrate
    # Group 6: Total Retransmits
    summary_pattern = re.compile(
        "\[\s*(SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)"
    )

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Try to parse interval data for individual streams and SUM
            interval_match = interval_pattern.search(line)
            if interval_match:
                stream_id = interval_match.group(1)
                # interval_start = float(interval_match.group(2)) # Not used
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
                        cwnd *= 1024 # Convert MBytes to KBytes for consistency

                if stream_id == 'SUM':
                    data['time'].append(interval_end)
                    data['sum_bitrate'].append(bitrate_val)
                    data['sum_retr'].append(retransmits)
                elif stream_id.isdigit():
                    data[f'stream{stream_id}_bitrate'].append(bitrate_val)
                    data[f'stream{stream_id}_retr'].append(retransmits)
                    data[f'stream{stream_id}_cwnd'].append(cwnd)

            # Try to parse summary line (final results)
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

    # Regex for interval lines: [ ID] Interval Transfer Bitrate Total Datagrams
    # Example: [  6]   0.00-1.00   sec   377 KBytes  3.08 Mbits/sec  482
    interval_pattern = re.compile(
        r"^\[\s*(\d+|SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+(KBytes|MBytes|GBytes)\s+([\d\.]+)\s+(Kbits/sec|Mbits/sec|Gbits/sec)\s+(\d+)\s*$"
    )
    
    # Regex for summary line: [SUM] 0.00-60.00 sec 224 MBytes 31.3 Mbits/sec 0.420 ms 107/293304 (0.036%) receiver
    summary_pattern = re.compile(
        r"^\[\s*(SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+([\d\.]+)\s+ms\s+(\d+)/(\d+)\s+\(([\d\.]+)%\)\s+receiver\s*$"
    )

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Try to parse interval data
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
                    data['sum_lost_datagrams'].append(0)  # Not available in interval lines
                    data['sum_total_datagrams'].append(total_datagrams)
                    data['sum_jitter'].append(0.0)  # Not available in interval lines
                elif stream_id.isdigit():
                    data[f'stream{stream_id}_transfer'].append(transfer_val)
                    data[f'stream{stream_id}_bitrate'].append(bitrate_val)
                    data[f'stream{stream_id}_datagrams'].append(total_datagrams)
                    data[f'stream{stream_id}_lost_datagrams'].append(0)  # Not available in interval lines
                    data[f'stream{stream_id}_total_datagrams'].append(total_datagrams)
                    data[f'stream{stream_id}_jitter'].append(0.0)  # Not available in interval lines

            # Try to parse summary line (final results)
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
    qdisc_pattern = re.compile(r"qdisc cake .*? bandwidth (\d+)(Kbit|Mbit|Gbit)")
    sent_dropped_pattern = re.compile(r"Sent (\d+) bytes (\d+) pkt \(dropped (\d+), overlimits (\d+) requeues (\d+)\)")
    memory_pattern = re.compile(r"memory used: (\d+)b of (\d+)(K|M|G)b")
    capacity_pattern = re.compile(r"capacity estimate: (\d+)(Kbit|Mbit|Gbit)")
    
    # Helper to convert units to a standard
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
                if current_stats:  # Save previous block if exists
                    intervals.append(current_stats.copy())
                current_stats = {'time': timestamp_match.group(1)}
                i += 1
                continue

            # Parse qdisc cake line (bandwidth, sent, dropped, overlimits, requeues)
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

            # Parse traffic class data (Bulk, Best Effort, Video, Voice)
            if "Bulk  Best Effort        Video        Voice" in line:  # Header for traffic classes
                headers = ["Bulk", "Best_Effort", "Video", "Voice"]
                j = i + 1
                while j < len(lines) and lines[j].strip() and not timestamp_pattern.search(lines[j]):
                    tc_line = lines[j].strip()
                    
                    # Parse pk_delay row
                    if "pk_delay" in tc_line:
                        pk_delay_match = re.search(r"pk_delay\s+(\d+)(us|ms)\s+(\d+)(us|ms)\s+(\d+)(us|ms)\s+(\d+)(us|ms)", tc_line)
                        if pk_delay_match:
                            current_stats['pk_delay_be'] = convert_unit(float(pk_delay_match.group(3)), pk_delay_match.group(4))
                            current_stats['pk_delay_video'] = convert_unit(float(pk_delay_match.group(5)), pk_delay_match.group(6))
                            current_stats['pk_delay_voice'] = convert_unit(float(pk_delay_match.group(7)), pk_delay_match.group(8))
                    
                    # Parse av_delay row
                    elif "av_delay" in tc_line:
                        av_delay_match = re.search(r"av_delay\s+(\d+)(us|ms)?\s+(\d+)(us|ms)?\s+(\d+)(us|ms)?\s+(\d+)(us|ms)?", tc_line)
                        if av_delay_match:
                            current_stats['av_delay_be'] = convert_unit(float(av_delay_match.group(3)), av_delay_match.group(4) if av_delay_match.group(4) else 'us')
                            current_stats['av_delay_video'] = convert_unit(float(av_delay_match.group(5)), av_delay_match.group(6) if av_delay_match.group(6) else 'us')
                            current_stats['av_delay_voice'] = convert_unit(float(av_delay_match.group(7)), av_delay_match.group(8) if av_delay_match.group(8) else 'us')
                    
                    # Parse drops row
                    elif "drops" in tc_line:
                        drops_match = re.search(r"drops\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", tc_line)
                        if drops_match:
                            current_stats['drops_be'] = int(drops_match.group(2))
                            current_stats['drops_video'] = int(drops_match.group(3))
                            current_stats['drops_voice'] = int(drops_match.group(4))
                    
                    # Parse marks row
                    elif "marks" in tc_line:
                        marks_match = re.search(r"marks\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", tc_line)
                        if marks_match:
                            current_stats['marks_be'] = int(marks_match.group(2))
                            current_stats['marks_video'] = int(marks_match.group(3))
                            current_stats['marks_voice'] = int(marks_match.group(4))
                    
                    # Parse pkts row
                    elif "pkts" in tc_line:
                        pkts_match = re.search(r"pkts\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", tc_line)
                        if pkts_match:
                            current_stats['pkts_be'] = int(pkts_match.group(2))
                            current_stats['pkts_video'] = int(pkts_match.group(3))
                            current_stats['pkts_voice'] = int(pkts_match.group(4))
                    
                    j += 1
                i = j - 1  # Adjust outer loop index
            i += 1

    if current_stats:  # Add the last block
        intervals.append(current_stats.copy())

    return intervals


# --- Phân tích chung ---
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


# --- Vẽ biểu đồ (TCP) ---
def plot_cake_analysis_tcp(data, final_stats, ping_times, out_img):
    n_plots = 6 if ping_times else 5  # Adjusted for potential ping plot
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    fig.suptitle('Phân tích hiệu năng CAKE TCP', fontsize=16, fontweight='bold')

    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]

    # Đảm bảo tất cả các stream có cùng độ dài
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

    # Plot 1: Bitrate
    ax = axs[0]
    if time_data and any(len(sb) > 0 for sb in stream_bitrates) and len(sum_bitrate) > 0:
        plotted_labels = set()
        for i, stream_bitrate in enumerate(stream_bitrates):
            if len(stream_bitrate) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, stream_bitrate, label=labels[i])
                plotted_labels.add(labels[i])
        if 'Tổng' not in plotted_labels:
            ax.plot(time_data, sum_bitrate, 'k--', label='Tổng', linewidth=2)
            plotted_labels.add('Tổng')
        # Ensure only 4 streams + 1 total are present
        allowed_labels = set(labels + ['Tổng'])
        seen = set()
        for line in list(ax.get_lines()):
            lbl = line.get_label()
            if lbl not in allowed_labels or lbl in seen:
                line.remove()
            else:
                seen.add(lbl)
        ax.set_title('Bitrate TCP theo thời gian')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Bitrate (Mbits/sec)')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu Bitrate TCP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Bitrate (Mbits/sec)')

    # Plot 2: Retransmissions
    ax = axs[1]
    if time_data and any(len(sr) > 0 for sr in stream_retrs) and len(sum_retr) > 0:
        plotted_labels = set()
        for i, stream_retr in enumerate(stream_retrs):
            if len(stream_retr) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, stream_retr, label=labels[i])
                plotted_labels.add(labels[i])
        if 'Tổng' not in plotted_labels:
            ax.plot(time_data, sum_retr, 'k--', label='Tổng', linewidth=2)
            plotted_labels.add('Tổng')
        # Ensure only 4 streams + 1 total are present
        allowed_labels = set(labels + ['Tổng'])
        seen = set()
        for line in list(ax.get_lines()):
            lbl = line.get_label()
            if lbl not in allowed_labels or lbl in seen:
                line.remove()
            else:
                seen.add(lbl)
        ax.set_title('Gói truyền lại TCP theo thời gian')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Số gói truyền lại')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu gói truyền lại TCP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Số gói truyền lại')

    # Plot 3: Congestion Window
    ax = axs[2]
    if time_data and any(len(sc) > 0 for sc in stream_cwnd):
        plotted_labels = set()
        for i, stream_c in enumerate(stream_cwnd):
            if len(stream_c) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, stream_c, label=labels[i])
                plotted_labels.add(labels[i])
        # Ensure only 4 streams are present (no total for cwnd)
        allowed_labels = set(labels)
        seen = set()
        for line in list(ax.get_lines()):
            lbl = line.get_label()
            if lbl not in allowed_labels or lbl in seen:
                line.remove()
            else:
                seen.add(lbl)
        ax.set_title('Cửa sổ tắc nghẽn (Cwnd) TCP theo thời gian')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Cwnd (KBytes)')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu Cwnd TCP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Cwnd (KBytes)')

    # Plot 4: Ping Latency (if available)
    if ping_times and len(axs) > 3:
        ax = axs[3]
        ax.plot(range(len(ping_times)), ping_times, 'b-')
        ax.set_title('Độ trễ Ping theo thời gian')
        ax.set_xlabel('Số thứ tự Ping')
        ax.set_ylabel('Độ trễ (ms)')
        ax.grid(True)
    elif len(axs) > 3:  # If ping_times is not available but subplot exists
        ax = axs[3]
        ax.set_title('Không có dữ liệu độ trễ Ping để hiển thị')
        ax.set_xlabel('Số thứ tự Ping')
        ax.set_ylabel('Độ trễ (ms)')

    # Plot 5: Bitrate Distribution (fairness)
    ax = axs[n_plots - 2]  # This will be either axs[3] or axs[4] depending on ping_times
    total_bitrates = [data.get(f'stream{i}_bitrate', []) for i in [6, 8, 10, 12]]
    flattened_bitrates = [item for sublist in total_bitrates for item in sublist if item is not None]
    
    if flattened_bitrates:
        ax.hist(flattened_bitrates, bins=20, edgecolor='black')
        ax.set_title('Phân bố Bitrate TCP giữa các luồng')
        ax.set_xlabel('Bitrate (Mbits/sec)')
        ax.set_ylabel('Tần suất')
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu phân bố Bitrate TCP để hiển thị')
        ax.set_xlabel('Bitrate (Mbits/sec)')
        ax.set_ylabel('Tần suất')

    # Plot 6: TCP Cumulative Transfer
    ax = axs[n_plots - 1]  # This will be either axs[4] or axs[5] depending on ping_times
    
    # Calculate cumulative transfer from bitrate data (since we don't have direct transfer data)
    if time_data and any(stream_bitrates):
        cumulative_transfer = {'stream6': [], 'stream8': [], 'stream10': [], 'stream12': []}
        
        for stream_idx, stream in enumerate(streams):
            stream_bitrate = stream_bitrates[stream_idx]
            if stream_bitrate:
                cumulative = 0
                for bitrate in stream_bitrate:
                    # Convert Mbits/sec to MBytes (assuming 1 second intervals)
                    # 1 Mbit = 0.125 MBytes
                    cumulative += (bitrate * 0.125)
                    cumulative_transfer[stream].append(cumulative)
            else:
                cumulative_transfer[stream] = [0] * len(time_data)

        # Use min_length for cumulative transfer as well
        min_length_transfer = min(len(time_data), *(len(v) for v in cumulative_transfer.values())) if time_data else 0
        time_data_transfer = time_data[:min_length_transfer]

        if time_data_transfer and any(cumulative_transfer.values()):
            for stream in streams:
                if cumulative_transfer[stream]:
                    ax.plot(time_data_transfer, cumulative_transfer[stream][:min_length_transfer], label=f'Luồng {stream[-2:]}')
            ax.set_title('Tổng Transfer TCP theo thời gian (tính từ Bitrate)')
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Transfer (MBytes)')
            ax.legend()
            ax.grid(True)
        else:
            ax.set_title('Không có dữ liệu Cumulative Transfer TCP để hiển thị')
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Transfer (MBytes)')
    else:
        ax.set_title('Không có dữ liệu Cumulative Transfer TCP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Transfer (MBytes)')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Add overall summary as text below the plots
    summary_text = f"""
    --- TCP Summary ---
    Tổng Bitrate: {final_stats.get('total_bitrate_tcp', 0.0):.2f} Mbits/sec
    Hiệu suất: {final_stats.get('efficiency_tcp', 0.0):.2f}%
    Tổng gói truyền lại: {final_stats.get('total_retr_tcp', 0)}
    Độ công bằng (Jain\'s Index): {final_stats.get('jains_index_iperf_tcp', 0.0):.3f}
    Min/Avg/Max/Std Latency (Ping): {final_stats.get('latency_tcp', {}).get('min', 0.0):.2f}/{final_stats.get('latency_tcp', {}).get('avg', 0.0):.2f}/{final_stats.get('latency_tcp', {}).get('max', 0.0):.2f}/{final_stats.get('latency_tcp', {}).get('std', 0.0):.2f} ms
    """
    plt.figtext(0.02, 0.01, summary_text, horizontalalignment='left', 
                verticalalignment='bottom', fontsize=10, bbox=dict(facecolor='wheat', alpha=0.5))
    
    plt.savefig(out_img)
    plt.close()

# --- Vẽ biểu đồ (UDP) ---
def plot_cake_analysis_udp(data, final_stats, ping_times, out_img):
    n_plots = 6 if ping_times else 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    fig.suptitle('Phân tích hiệu năng CAKE UDP', fontsize=16, fontweight='bold')

    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]

    min_length = min(
        len(data.get('time', [])),
        *(len(data.get(f'stream{i}_bitrate', [])) for i in [6, 8, 10, 12]),
        len(data.get('sum_bitrate', []))
    ) if data.get('time') else 0
    
    time_data = data.get('time', [])[:min_length] if min_length > 0 else []
    stream_bitrates = [data.get(f'stream{i}_bitrate', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    sum_bitrate = data.get('sum_bitrate', [])[:min_length] if min_length > 0 else []
    stream_jitt_raw = [data.get(f'stream{i}_jitter', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    sum_jitt = data.get('sum_jitter', [])[:min_length] if min_length > 0 else []
    stream_lost_pkt = [data.get(f'stream{i}_lost_datagrams', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    stream_total_pkt = [data.get(f'stream{i}_total_datagrams', [])[:min_length] for i in [6, 8, 10, 12]] if min_length > 0 else [[], [], [], []]
    sum_lost_pkt = data.get('sum_lost_datagrams', [])[:min_length] if min_length > 0 else []
    sum_total_pkt = data.get('sum_total_datagrams', [])[:min_length] if min_length > 0 else []

    # Plot 1: Bitrate
    ax = axs[0]
    if time_data and any(len(sb) > 0 for sb in stream_bitrates) and len(sum_bitrate) > 0:
        plotted_labels = set()
        for i, stream_bitrate in enumerate(stream_bitrates):
            if len(stream_bitrate) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, stream_bitrate, label=labels[i])
                plotted_labels.add(labels[i])
        if 'Tổng' not in plotted_labels:
            ax.plot(time_data, sum_bitrate, 'k--', label='Tổng', linewidth=2)
            plotted_labels.add('Tổng')
        # Ensure only 4 streams + 1 total are present
        allowed_labels = set(labels + ['Tổng'])
        seen = set()
        for line in list(ax.get_lines()):
            lbl = line.get_label()
            if lbl not in allowed_labels or lbl in seen:
                line.remove()
            else:
                seen.add(lbl)
        ax.set_title('Bitrate UDP theo thời gian')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Bitrate (Mbits/sec)')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu Bitrate UDP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Bitrate (Mbits/sec)')

    # Plot 2: Jitter
    ax = axs[1]
    if time_data and any(len(sj) > 0 for sj in stream_jitt_raw) and len(sum_jitt) > 0:
        plotted_labels = set()
        for i, stream_jitt in enumerate(stream_jitt_raw):
            if len(stream_jitt) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, stream_jitt, label=labels[i])
                plotted_labels.add(labels[i])
        if 'Tổng' not in plotted_labels:
            ax.plot(time_data, sum_jitt, 'k--', label='Tổng', linewidth=2)
            plotted_labels.add('Tổng')
        # Ensure only 4 streams + 1 total are present
        allowed_labels = set(labels + ['Tổng'])
        seen = set()
        for line in list(ax.get_lines()):
            lbl = line.get_label()
            if lbl not in allowed_labels or lbl in seen:
                line.remove()
            else:
                seen.add(lbl)
        ax.set_title('Jitter UDP theo thời gian')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Jitter (ms)')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu Jitter UDP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Jitter (ms)')

    # Plot 3: Lost Datagrams
    ax = axs[2]
    # Calculate loss rate per interval
    loss_rates = []
    if time_data and any(stream_total_pkt):
        for i in range(min_length):
            lost = sum(s[i] for s in stream_lost_pkt if i < len(s))
            total = sum(s[i] for s in stream_total_pkt if i < len(s))
            if total > 0:  # Check to prevent ZeroDivisionError
                loss_rates.append((lost / total) * 100)
            else:
                loss_rates.append(0)

    if time_data and loss_rates:
        ax.plot(time_data, loss_rates, 'r-', label='Tỷ lệ mất gói', linewidth=2)
        ax.set_title('Tỷ lệ mất gói UDP theo thời gian')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Tỷ lệ mất gói (%)')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Không có dữ liệu mất gói UDP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Tỷ lệ mất gói (%)')

    # Plot 4: Ping Latency (if available)
    if ping_times and len(axs) > 3:
        ax = axs[3]
        ax.plot(range(len(ping_times)), ping_times, 'b-')
        ax.set_title('Độ trễ Ping theo thời gian')
        ax.set_xlabel('Số thứ tự Ping')
        ax.set_ylabel('Độ trễ (ms)')
        ax.grid(True)
    elif len(axs) > 3:
        ax = axs[3]
        ax.set_title('Không có dữ liệu độ trễ Ping để hiển thị')
        ax.set_xlabel('Số thứ tự Ping')
        ax.set_ylabel('Độ trễ (ms)')

    # Plot 5: UDP Cumulative Transfer
    ax = axs[n_plots - 1]  # This will be either axs[3] or axs[4] depending on ping_times
    
    # Calculate cumulative transfer from bitrate data (since we don't have direct transfer data)
    if time_data and any(stream_bitrates):
        cumulative_transfer = {'stream6': [], 'stream8': [], 'stream10': [], 'stream12': []}
        
        for stream_idx, stream in enumerate(streams):
            stream_bitrate = stream_bitrates[stream_idx]
            if stream_bitrate:
                cumulative = 0
                for bitrate in stream_bitrate:
                    # Convert Mbits/sec to MBytes (assuming 1 second intervals)
                    # 1 Mbit = 0.125 MBytes
                    cumulative += (bitrate * 0.125)
                    cumulative_transfer[stream].append(cumulative)
            else:
                cumulative_transfer[stream] = [0] * len(time_data)

        # Use min_length for cumulative transfer as well
        min_length_transfer = min(len(time_data), *(len(v) for v in cumulative_transfer.values())) if time_data else 0
        time_data_transfer = time_data[:min_length_transfer]

        if time_data_transfer and any(cumulative_transfer.values()):
            for stream in streams:
                if cumulative_transfer[stream]:
                    ax.plot(time_data_transfer, cumulative_transfer[stream][:min_length_transfer], label=f'Luồng {stream[-2:]}')
            ax.set_title('Tổng Transfer UDP theo thời gian (tính từ Bitrate)')
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Transfer (MBytes)')
            ax.legend()
            ax.grid(True)
        else:
            ax.set_title('Không có dữ liệu Cumulative Transfer UDP để hiển thị')
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Transfer (MBytes)')
    else:
        ax.set_title('Không có dữ liệu Cumulative Transfer UDP để hiển thị')
        ax.set_xlabel('Thời gian (giây)')
        ax.set_ylabel('Transfer (MBytes)')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    summary_text = f"""
    --- UDP Summary ---
    Tổng Bitrate: {final_stats.get('total_bitrate_udp', 0.0):.2f} Mbits/sec
    Hiệu suất: {final_stats.get('efficiency_udp', 0.0):.2f}%
    Jitter TB: {final_stats.get('avg_jitter_udp', 0.0):.2f} ms
    Gói mất/Tổng gói: {final_stats.get('lost_datagrams_udp', 0)}/{final_stats.get('total_datagrams_udp', 0)}
    Tỷ lệ mất gói: {final_stats.get('loss_percent_udp', 0.0):.2f}%
    Độ công bằng (Jain\'s Index): {final_stats.get('jains_index_iperf_udp', 0.0):.3f}
    Min/Avg/Max/Std Latency (Ping): {final_stats.get('latency_udp', {}).get('min', 0.0):.2f}/{final_stats.get('latency_udp', {}).get('avg', 0.0):.2f}/{final_stats.get('latency_udp', {}).get('max', 0.0):.2f}/{final_stats.get('latency_udp', {}).get('std', 0.0):.2f} ms
    """
    plt.figtext(0.02, 0.01, summary_text, horizontalalignment='left', 
                verticalalignment='bottom', fontsize=10, bbox=dict(facecolor='wheat', alpha=0.5))
    
    plt.savefig(out_img)
    plt.close()

# --- Vẽ biểu đồ (CAKE Stats) ---
def plot_cake_stats_analysis(intervals, out_img):
    if not intervals:
        print("Không có dữ liệu thống kê CAKE để vẽ biểu đồ.")
        return

    # Extract data for plotting
    times = list(range(len(intervals)))  # Use numeric index instead of datetime conversion
    dropped_pkts = [interval.get('dropped_pkt', 0) for interval in intervals]
    overlimits = [interval.get('overlimits', 0) for interval in intervals]
    requeues = [interval.get('requeues', 0) for interval in intervals]
    
    # Delays for Best Effort, Video, Voice
    pk_delay_be = [interval.get('pk_delay_be', 0) for interval in intervals]
    pk_delay_video = [interval.get('pk_delay_video', 0) for interval in intervals]
    pk_delay_voice = [interval.get('pk_delay_voice', 0) for interval in intervals]

    av_delay_be = [interval.get('av_delay_be', 0) for interval in intervals]
    av_delay_video = [interval.get('av_delay_video', 0) for interval in intervals]
    av_delay_voice = [interval.get('av_delay_voice', 0) for interval in intervals]

    # Drops and Marks per traffic class
    drops_be = [interval.get('drops_be', 0) for interval in intervals]
    drops_video = [interval.get('drops_video', 0) for interval in intervals]
    drops_voice = [interval.get('drops_voice', 0) for interval in intervals]

    marks_be = [interval.get('marks_be', 0) for interval in intervals]
    marks_video = [interval.get('marks_video', 0) for interval in intervals]
    marks_voice = [interval.get('marks_voice', 0) for interval in intervals]

    num_plots = 3 # Base plots: Dropped, Overlimits, Requeues
    if any(pk_delay_be) or any(av_delay_be):
        num_plots += 1
    if any(drops_be) or any(drops_video) or any(drops_voice):
        num_plots += 1
    if any(marks_be) or any(marks_video) or any(marks_voice):
        num_plots += 1

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    fig.suptitle('Phân tích thống kê CAKE', fontsize=16, fontweight='bold')

    current_plot_idx = 0

    # Plot 1: Dropped Packets Over Time
    ax = axs[current_plot_idx]
    if times and dropped_pkts:
        ax.plot(times, dropped_pkts, 'r-', label='Gói bị loại')
        ax.set_title('Gói bị loại theo thời gian')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số gói')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.set_title('Không có dữ liệu gói bị loại để hiển thị')
    current_plot_idx += 1

    # Plot 2: Overlimits Over Time
    ax = axs[current_plot_idx]
    if times and overlimits:
        ax.plot(times, overlimits, 'g-', label='Overlimits')
        ax.set_title('Overlimits theo thời gian')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số lần')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.set_title('Không có dữ liệu Overlimits để hiển thị')
    current_plot_idx += 1

    # Plot 3: Requeues Over Time
    ax = axs[current_plot_idx]
    if times and requeues:
        ax.plot(times, requeues, 'b-', label='Requeues')
        ax.set_title('Requeues theo thời gian')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số lần')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.set_title('Không có dữ liệu Requeues để hiển thị')
    current_plot_idx += 1

    # Plot 4: Peak and Average Delays for Traffic Classes
    if any(pk_delay_be) or any(av_delay_be):
        ax = axs[current_plot_idx]
        if times and pk_delay_be and pk_delay_video and pk_delay_voice:
            ax.plot(times, pk_delay_be, label='Pk Delay (Best Effort)', linestyle='--')
            ax.plot(times, pk_delay_video, label='Pk Delay (Video)', linestyle='--')
            ax.plot(times, pk_delay_voice, label='Pk Delay (Voice)', linestyle='--')
        if times and av_delay_be and av_delay_video and av_delay_voice:
            ax.plot(times, av_delay_be, label='Av Delay (Best Effort)', linestyle='-')
            ax.plot(times, av_delay_video, label='Av Delay (Video)', linestyle='-')
            ax.plot(times, av_delay_voice, label='Av Delay (Voice)', linestyle='-')
        
        ax.set_title('Độ trễ trung bình và đỉnh theo thời gian (Traffic Classes)')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Độ trễ (ms)')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        current_plot_idx += 1
    else:
        # Check if this subplot was allocated but no data
        if num_plots > current_plot_idx:
            ax = axs[current_plot_idx]
            ax.set_title('Không có dữ liệu độ trễ cho Traffic Classes để hiển thị')
            current_plot_idx += 1


    # Plot 5: Drops per Traffic Class
    if any(drops_be) or any(drops_video) or any(drops_voice):
        ax = axs[current_plot_idx]
        if times and drops_be and drops_video and drops_voice:
            ax.plot(times, drops_be, label='Drops (Best Effort)')
            ax.plot(times, drops_video, label='Drops (Video)')
            ax.plot(times, drops_voice, label='Drops (Voice)')
        ax.set_title('Gói bị loại theo lớp lưu lượng')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số gói bị loại')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        current_plot_idx += 1
    else:
        if num_plots > current_plot_idx:
            ax = axs[current_plot_idx]
            ax.set_title('Không có dữ liệu gói bị loại theo lớp lưu lượng để hiển thị')
            current_plot_idx += 1

    # Plot 6: Marks per Traffic Class
    if any(marks_be) or any(marks_video) or any(marks_voice):
        ax = axs[current_plot_idx]
        if times and marks_be and marks_video and marks_voice:
            ax.plot(times, marks_be, label='Marks (Best Effort)')
            ax.plot(times, marks_video, label='Marks (Video)')
            ax.plot(times, marks_voice, label='Marks (Voice)')
        ax.set_title('Gói được đánh dấu ECN theo lớp lưu lượng')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Số gói được đánh dấu')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        current_plot_idx += 1
    else:
        if num_plots > current_plot_idx:
            ax = axs[current_plot_idx]
            ax.set_title('Không có dữ liệu gói được đánh dấu ECN theo lớp lưu lượng để hiển thị')
            current_plot_idx += 1

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent overlap
    plt.savefig(out_img)
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

    # Ping Latency Metrics (from TCP analysis, assuming same ping for both)
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
    iperf_file_udp = r'd:/Ths/20252/v2/cake/iperf_cake_udp.log'
    iperf_file_tcp = r'd:/Ths/20252/v2/cake/iperf_cake_tcp.log'
    ping_file = r'd:/Ths/20252/v2/cake/ping_cake.log'
    cake_stats_file = r'd:/Ths/20252/v2/cake/cake_stats.txt'
    target_bandwidth = 100  # Mbps (đặt mục tiêu băng thông)

    print("="*80)
    print("PHÂN TÍCH HIỆU NĂNG THUẬT TOÁN CAKE")
    print("="*80)

    print("\nPhân tích file iperf_cake_udp.log (UDP)...")
    data_udp, final_stats_udp = parse_iperf_log_udp(iperf_file_udp)

    print("\nPhân tích file iperf_cake_tcp.log (TCP)...")
    data_tcp, final_stats_tcp = parse_iperf_log(iperf_file_tcp)

    print("\nPhân tích file ping_cake.log ...")
    ping_times = parse_ping_log(ping_file)

    print("\nPhân tích file cake_stats.txt ...")
    cake_stats_intervals = parse_cake_stats(cake_stats_file)

    # Phân tích và in kết quả TCP
    print("\n--- Kết quả phân tích TCP ---")
    if 'total_bitrate_tcp' in final_stats_tcp:
        efficiency_tcp = analyze_bandwidth(final_stats_tcp['total_bitrate_tcp'], target_bandwidth)
        final_stats_tcp['efficiency_tcp'] = efficiency_tcp
        print(f"Tổng băng thông TCP: {final_stats_tcp['total_bitrate_tcp']:.2f} Mbits/sec (Hiệu suất: {efficiency_tcp:.2f}%)")
        print(f"Tổng gói truyền lại TCP: {final_stats_tcp.get('total_retr_tcp', 0)}")
    
    # Get all stream bitrates for fairness calculation
    all_tcp_stream_bitrates = []
    for stream_id in [6, 8, 10, 12]:
        all_tcp_stream_bitrates.extend(data_tcp.get(f'stream{stream_id}_bitrate', []))

    tcp_fairness = analyze_fairness_wrapper('tcp', stream_bitrates=all_tcp_stream_bitrates)
    final_stats_tcp.update(tcp_fairness)
    print(f"Độ công bằng (Jain\'s Index) TCP: {final_stats_tcp.get('jains_index_iperf_tcp', 0.0):.3f}")

    tcp_latency_stats = analyze_latency(ping_times) # Assuming ping is generally for overall network latency impacting both
    final_stats_tcp['latency_tcp'] = tcp_latency_stats
    print(f"Độ trễ Ping (Min/Avg/Max/Std): {tcp_latency_stats['min']:.2f}/{tcp_latency_stats['avg']:.2f}/{tcp_latency_stats['max']:.2f}/{tcp_latency_stats['std']:.2f} ms")

    # Phân tích và in kết quả UDP
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

    udp_latency_stats = analyze_latency(ping_times) # Assuming same ping for both
    final_stats_udp['latency_udp'] = udp_latency_stats
    print(f"Độ trễ Ping (Min/Avg/Max/Std): {udp_latency_stats['min']:.2f}/{udp_latency_stats['avg']:.2f}/{udp_latency_stats['max']:.2f}/{udp_latency_stats['std']:.2f} ms")

    # Phân tích và in kết quả CAKE Stats
    print("\n--- Kết quả phân tích CAKE Stats ---")
    if cake_stats_intervals:
        total_dropped_from_stats = sum(interval.get('dropped_pkt', 0) for interval in cake_stats_intervals)
        total_sent_from_stats = sum(interval.get('sent_pkt', 0) for interval in cake_stats_intervals)
        loss_rate_stats = analyze_packet_loss(total_sent_from_stats, total_dropped_from_stats)
        print(f"Tổng gói Sent (từ cake_stats): {total_sent_from_stats}, Tổng gói bị loại (từ cake_stats): {total_dropped_from_stats}, Tỷ lệ mất gói: {loss_rate_stats:.2f}%")

        # Example of extracting overall average delays for traffic classes
        avg_pk_delay_be = np.mean([interval.get('pk_delay_be', 0) for interval in cake_stats_intervals if interval.get('pk_delay_be') is not None])
        avg_av_delay_be = np.mean([interval.get('av_delay_be', 0) for interval in cake_stats_intervals if interval.get('av_delay_be') is not None])
        print(f"Avg Peak Delay (Best Effort): {avg_pk_delay_be:.2f} ms")
        print(f"Avg Average Delay (Best Effort): {avg_av_delay_be:.2f} ms")
    else:
        print("Không có dữ liệu thống kê CAKE để phân tích.")

    # Vẽ biểu đồ và lưu ảnh
    print("\nVẽ biểu đồ và lưu ảnh...")
    plot_cake_analysis_tcp(data_tcp, final_stats_tcp, ping_times, 'cake_performance_analysis_tcp.png')
    plot_cake_analysis_udp(data_udp, final_stats_udp, ping_times, 'cake_performance_analysis_udp.png')
    plot_cake_stats_analysis(cake_stats_intervals, 'cake_stats_analysis.png')

    # Xuất báo cáo CSV
    print("\nXuất báo cáo CSV...")
    combined_final_stats = {**final_stats_tcp, **final_stats_udp}
    export_cake_report(combined_final_stats, 'cake_performance_report.csv')

    print("\nQuá trình phân tích hiệu năng CAKE hoàn tất.") 