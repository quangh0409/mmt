import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Phân tích iperf_red.log (TCP) ---
def parse_iperf_log_tcp(filename):
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

    # Interval line regex (with Retr and Cwnd)
    interval_pattern = re.compile(
        r'\[\s*(\d+)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)\s+([\d\.]+)\s+KBytes'
    )
    sum_pattern = re.compile(
        r'\[SUM\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)'
    )
    # Final summary regex
    final_stream_pattern = re.compile(
        r'\[\s*(\d+)\]\s+0\.00-60\.00\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)'
    )
    final_sum_pattern = re.compile(
        r'\[SUM\]\s+0\.00-60\.00\s+sec\s+([\d\.]+)\s+GBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)'
    )

    with open(filename, encoding='utf-8') as f:
        for line in f:
            interval_match = interval_pattern.match(line)
            if interval_match:
                stream_id, start, end, transfer, bitrate, retr, cwnd = interval_match.groups()
                if float(start) not in data['time']:
                    data['time'].append(float(start))
                bitrate = float(bitrate)
                retr = int(retr)
                cwnd = float(cwnd)
                if stream_id == '6':
                    data['stream6_bitrate'].append(bitrate)
                    data['stream6_retr'].append(retr)
                    data['stream6_cwnd'].append(cwnd)
                elif stream_id == '8':
                    data['stream8_bitrate'].append(bitrate)
                    data['stream8_retr'].append(retr)
                    data['stream8_cwnd'].append(cwnd)
                elif stream_id == '10':
                    data['stream10_bitrate'].append(bitrate)
                    data['stream10_retr'].append(retr)
                    data['stream10_cwnd'].append(cwnd)
                elif stream_id == '12':
                    data['stream12_bitrate'].append(bitrate)
                    data['stream12_retr'].append(retr)
                    data['stream12_cwnd'].append(cwnd)
            sum_match = sum_pattern.match(line)
            if sum_match:
                start, end, transfer, bitrate, retr = sum_match.groups()
                data['sum_bitrate'].append(float(bitrate))
                data['sum_retr'].append(int(retr))
            final_stream_match = final_stream_pattern.match(line)
            if final_stream_match:
                stream_id, transfer, bitrate, total_retr = final_stream_match.groups()
                final_stats[f'stream{stream_id}'] = {
                    'transfer': float(transfer),
                    'bitrate': float(bitrate),
                    'total_retr': int(total_retr)
                }
            final_sum_match = final_sum_pattern.match(line)
            if final_sum_match:
                transfer, bitrate, total_retr = final_sum_match.groups()
                final_stats['total_transfer'] = float(transfer) * 1024  # GBytes to MBytes
                final_stats['total_bitrate'] = float(bitrate)
                final_stats['total_retr'] = int(total_retr)

    return data, final_stats

# --- Phân tích iperf_red_udp.log (UDP) ---
def parse_iperf_log_udp(filename):
    data = {
        'time': [],
        'stream6_bitrate': [],
        'stream8_bitrate': [],
        'stream10_bitrate': [],
        'stream12_bitrate': [],
        'sum_bitrate': [],
        'stream6_jitter': [],
        'stream8_jitter': [],
        'stream10_jitter': [],
        'stream12_jitter': [],
        'stream6_lost': [],
        'stream8_lost': [],
        'stream10_lost': [],
        'stream12_lost': [],
        'stream6_total': [],
        'stream8_total': [],
        'stream10_total': [],
        'stream12_total': [],
    }
    final_stats = {}

    # Interval line regex (bitrate)
    interval_pattern = re.compile(
        r'\[\s*(\d+)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+KBytes\s+([\d\.]+)\s+Mbits/sec\s+\d+'
    )
    sum_pattern = re.compile(
        r'\[SUM\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+\d+'
    )
    # Final summary regex (jitter/lost/total)
    final_stream_pattern = re.compile(
        r'\[\s*(\d+)\]\s+0\.00-60\.00\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+([\d\.]+)\s+ms\s+(\d+)/(\d+)'
    )
    final_sum_pattern = re.compile(
        r'\[SUM\]\s+0\.00-60\.00\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+([\d\.]+)\s+ms\s+(\d+)/(\d+)'
    )

    with open(filename, encoding='utf-8') as f:
        for line in f:
            interval_match = interval_pattern.match(line)
            if interval_match:
                stream_id, start, end, transfer, bitrate = interval_match.groups()
                if float(start) not in data['time']:
                    data['time'].append(float(start))
                bitrate = float(bitrate)
                if stream_id == '6':
                    data['stream6_bitrate'].append(bitrate)
                elif stream_id == '8':
                    data['stream8_bitrate'].append(bitrate)
                elif stream_id == '10':
                    data['stream10_bitrate'].append(bitrate)
                elif stream_id == '12':
                    data['stream12_bitrate'].append(bitrate)
            sum_match = sum_pattern.match(line)
            if sum_match:
                start, end, transfer, bitrate = sum_match.groups()
                data['sum_bitrate'].append(float(bitrate))
            final_stream_match = final_stream_pattern.match(line)
            if final_stream_match:
                stream_id, transfer, bitrate, jitter, lost, total = final_stream_match.groups()
                final_stats[f'stream{stream_id}'] = {
                    'transfer': float(transfer),
                    'bitrate': float(bitrate),
                    'jitter': float(jitter),
                    'lost': int(lost),
                    'total': int(total)
                }
            final_sum_match = final_sum_pattern.match(line)
            if final_sum_match:
                transfer, bitrate, jitter, lost, total = final_sum_match.groups()
                final_stats['total_transfer'] = float(transfer)
                final_stats['total_bitrate'] = float(bitrate)
                final_stats['total_jitter'] = float(jitter)
                final_stats['total_lost'] = int(lost)
                final_stats['total_datagrams'] = int(total)

    return data, final_stats

# --- Phân tích ping_red.log ---
def parse_ping_log(filename):
    times = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'time=([\d\.]+) ms', line)
            if m:
                times.append(float(m.group(1)))
    return times

# --- Phân tích red_stats.txt ---
def parse_red_stats(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.read()
    sent_pkt = int(re.search(r'Sent.*?(\d+) pkt', text).group(1))
    dropped = int(re.search(r'dropped (\d+)', text).group(1))
    marked = int(re.search(r'marked (\d+)', text).group(1))
    return sent_pkt, dropped, marked

# --- Phân tích red_stats_interval.txt ---
def parse_red_stats_interval(filename):
    intervals = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            if 'qdisc red' in line:
                time_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
                if time_match:
                    time = time_match.group(1)
                    # Đọc dòng tiếp theo để lấy thống kê
                    next_line = next(f, '')
                    sent_match = re.search(r'Sent.*?(\d+) bytes (\d+) pkt', next_line)
                    if sent_match:
                        bytes_sent = int(sent_match.group(1))
                        pkts_sent = int(sent_match.group(2))
                        dropped_match = re.search(r'dropped (\d+)', next_line)
                        dropped = int(dropped_match.group(1)) if dropped_match else 0
                        marked_match = re.search(r'marked (\d+)', next_line)
                        marked = int(marked_match.group(1)) if marked_match else 0
                        intervals.append({
                            'time': time,
                            'bytes_sent': bytes_sent,
                            'pkts_sent': pkts_sent,
                            'dropped': dropped,
                            'marked': marked
                        })
    return intervals

# --- Phân tích hiệu suất ---
def analyze_bandwidth(final_stats, target_bandwidth):
    total_bitrate = final_stats.get('total_bitrate', 0)
    efficiency = (total_bitrate / target_bandwidth) * 100 if target_bandwidth else 0
    print(f"Băng thông trung bình: {total_bitrate:.2f} Mbps")
    print(f"Hiệu suất đạt được: {efficiency:.2f}% so với mục tiêu {target_bandwidth} Mbps")
    return total_bitrate, efficiency

def analyze_packet_loss(sent_pkt, dropped, marked=0):
    loss_rate = (dropped / sent_pkt) * 100 if sent_pkt else 0
    mark_rate = (marked / sent_pkt) * 100 if sent_pkt else 0
    print(f"Tổng gói gửi: {sent_pkt}, Gói bị loại: {dropped}, Gói bị mark: {marked}")
    print(f"Tỷ lệ mất gói: {loss_rate:.2f}%, Tỷ lệ mark: {mark_rate:.2f}%")
    return loss_rate, mark_rate

def analyze_latency(times):
    if not times:
        print("Không có dữ liệu ping.")
        return None, None, None, None
    avg = np.mean(times)
    minv = np.min(times)
    maxv = np.max(times)
    jitter = np.std(times)
    print(f"Độ trễ trung bình: {avg:.2f} ms, min: {minv:.2f} ms, max: {maxv:.2f} ms, jitter: {jitter:.2f} ms")
    return avg, minv, maxv, jitter

def analyze_fairness(final_stats, ping_times=None):
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    fairness = 0
    avg_bitrates = [final_stats[s]['bitrate'] if s in final_stats and 'bitrate' in final_stats[s] else 0 for s in streams]
    if any(avg_bitrates):
        fairness = (sum(avg_bitrates) ** 2) / (len(avg_bitrates) * sum(b ** 2 for b in avg_bitrates)) if max(avg_bitrates) > 0 else 0

    total_transfer = sum(final_stats[s]['transfer'] if s in final_stats and 'transfer' in final_stats[s] else 0 for s in streams)
    total_bitrate = sum(avg_bitrates)
    total_retr = final_stats.get('total_retr', 0)
    avg_jitter = np.mean([final_stats[s]['jitter'] for s in streams if s in final_stats and 'jitter' in final_stats[s]]) if any('jitter' in final_stats.get(s, {}) for s in streams) else None
    avg_loss = np.mean([100 * final_stats[s]['lost'] / final_stats[s]['total'] if s in final_stats and 'lost' in final_stats[s] and 'total' in final_stats[s] and final_stats[s]['total'] > 0 else 0 for s in streams]) if any('lost' in final_stats.get(s, {}) for s in streams) else None

    ping_avg = ping_min = ping_max = ping_jitter = None
    if ping_times:
        ping_avg = np.mean(ping_times)
        ping_min = np.min(ping_times)
        ping_max = np.max(ping_times)
        ping_jitter = np.std(ping_times)

    print(f"Chỉ số công bằng (Jain's fairness): {fairness:.3f}")
    return {
        'fairness': fairness,
        'total_transfer': total_transfer,
        'total_bitrate': total_bitrate,
        'total_retr': total_retr,
        'avg_jitter': avg_jitter,
        'avg_loss': avg_loss,
        'ping_avg': ping_avg,
        'ping_min': ping_min,
        'ping_max': ping_max,
        'ping_jitter': ping_jitter,
    }

# --- Vẽ biểu đồ ---
def plot_red_analysis_tcp(data, final_stats, ping_times, out_img):
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]
    min_len = min(
        len(data.get('time', [])),
        *(len(data.get(f'stream{i}_bitrate', [])) for i in [6, 8, 10, 12]),
        len(data.get('sum_bitrate', []))
    )
    time_points = data.get('time', [])[:min_len]
    stream_bitrates = [data.get(f'stream{i}_bitrate', [])[:min_len] for i in [6, 8, 10, 12]]
    sum_bitrate = data.get('sum_bitrate', [])[:min_len]
    stream_retrs = [data.get(f'stream{i}_retr', [])[:min_len] for i in [6, 8, 10, 12]]
    sum_retr = data.get('sum_retr', [])[:min_len]
    stream_cwnds = [data.get(f'stream{i}_cwnd', [])[:min_len] for i in [6, 8, 10, 12]]

    avg_bitrates = [final_stats[s]['bitrate'] if s in final_stats and 'bitrate' in final_stats[s] else 0 for s in streams]
    total_retr = final_stats.get('total_retr', 0)
    retr_vals = [final_stats[s]['total_retr'] if s in final_stats and 'total_retr' in final_stats[s] else 0 for s in streams]

    n_plots = 6 if ping_times else 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

    # 1. Bitrate theo thời gian
    ax = axs[0]
    for i, s_bitrate in enumerate(stream_bitrates):
        ax.plot(time_points, s_bitrate, label=labels[i])
    ax.plot(time_points, sum_bitrate, label='Tổng', linewidth=2, linestyle='--')
    ax.set_title('Bitrate theo thời gian (RED TCP)')
    ax.set_xlabel('Thời gian (s)')
    ax.set_ylabel('Bitrate (Mbits/sec)')
    ax.grid(True)
    ax.legend()

    # 2. Số gói truyền lại theo thời gian
    ax = axs[1]
    for i, s_retr in enumerate(stream_retrs):
        ax.plot(time_points, s_retr, label=labels[i])
    if sum_retr:
        ax.plot(time_points, sum_retr, label='Tổng', linewidth=2, linestyle='--')
    ax.set_title('Số gói truyền lại theo thời gian')
    ax.set_xlabel('Thời gian (s)')
    ax.set_ylabel('Retransmits')
    ax.grid(True)
    ax.legend()

    # 3. Tốc độ trung bình từng luồng
    ax = axs[2]
    ax.bar(labels, avg_bitrates, color=['blue', 'green', 'red', 'cyan'])
    ax.set_title('Tốc độ trung bình từng luồng')
    ax.set_ylabel('Bitrate (Mbits/sec)')
    ax.grid(True, axis='y')

    # 4. Tổng số gói truyền lại từng luồng và tổng
    ax = axs[3]
    bar_labels = labels + ['Tổng']
    bar_vals = retr_vals + [total_retr]
    ax.bar(bar_labels, bar_vals, color=['blue', 'green', 'red', 'cyan', 'orange'])
    ax.set_title('Tổng số gói truyền lại từng luồng và tổng')
    ax.set_ylabel('Retransmits')
    ax.grid(True, axis='y')

    # 5. Biểu đồ Cwnd theo thời gian
    ax = axs[4]
    for i, s_cwnd in enumerate(stream_cwnds):
        ax.plot(time_points, s_cwnd, label=labels[i])
    ax.set_title('Cwnd (KBytes) theo thời gian')
    ax.set_xlabel('Thời gian (s)')
    ax.set_ylabel('Cwnd (KBytes)')
    ax.grid(True)
    ax.legend()

    # 6. Biểu đồ Ping (nếu có)
    if ping_times:
        ax = axs[5]
        ax.plot(ping_times, marker='.', linestyle='-', color='orange')
        ax.set_title('Phân bố độ trễ Ping')
        ax.set_xlabel('Chỉ số gói')
        ax.set_ylabel('Độ trễ (ms)')
        ax.grid(True)

    # Thông số tổng hợp dưới biểu đồ
    streams_names = ['stream6', 'stream8', 'stream10', 'stream12']
    total_transfer = sum(final_stats[s]['transfer'] if s in final_stats and 'transfer' in final_stats[s] else 0 for s in streams_names)
    total_bitrate_sum = sum(avg_bitrates)
    fairness = 0
    if any(avg_bitrates) and max(avg_bitrates) > 0:
        fairness = (sum(avg_bitrates) ** 2) / (len(avg_bitrates) * sum(b ** 2 for b in avg_bitrates))
    ping_avg = ping_min = ping_max = ping_jitter = None
    if ping_times:
        ping_avg = np.mean(ping_times)
        ping_min = np.min(ping_times)
        ping_max = np.max(ping_times)
        ping_jitter = np.std(ping_times)
    stats_text = (
        f"Tổng dữ liệu truyền: {total_transfer:.1f} MB\n"
        f"Tốc độ tổng trung bình: {total_bitrate_sum:.1f} Mbps\n"
        f"Tổng số gói truyền lại: {total_retr}\n"
        f"Tỷ lệ công bằng (Jain): {fairness:.3f}\n"
        + (f"Ping trung bình: {ping_avg:.2f} ms, min: {ping_min:.2f} ms, max: {ping_max:.2f} ms, jitter: {ping_jitter:.2f} ms" if ping_avg is not None else "")
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12,
                bbox=dict(facecolor='lightgray', alpha=0.5))

    fig.suptitle("Phân tích Hiệu Năng Thuật Toán RED (TCP)", fontsize=18, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_img, dpi=300)
    plt.close(fig)

def plot_red_analysis_udp(final_stats, ping_times, out_img):
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]
    transfers = [final_stats[s]['transfer'] if s in final_stats and 'transfer' in final_stats[s] else 0 for s in streams]
    bitrates = [final_stats[s]['bitrate'] if s in final_stats and 'bitrate' in final_stats[s] else 0 for s in streams]
    jitters = [final_stats[s]['jitter'] if s in final_stats and 'jitter' in final_stats[s] else 0 for s in streams]
    losts = [final_stats[s]['lost'] if s in final_stats and 'lost' in final_stats[s] else 0 for s in streams]
    totals = [final_stats[s]['total'] if s in final_stats and 'total' in final_stats[s] else 1 for s in streams]
    loss_rates = [100 * lost / total if total > 0 else 0 for lost, total in zip(losts, totals)]

    n_plots = 5 if ping_times else 4
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

    # 1. Transfer
    ax = axs[0]
    ax.bar(labels, transfers, color='skyblue')
    ax.set_title('Transfer (MB) từng luồng (RED UDP)')
    ax.set_ylabel('MB')
    ax.grid(True, axis='y')

    # 2. Bitrate
    ax = axs[1]
    ax.bar(labels, bitrates, color='orange')
    ax.set_title('Bitrate (Mbps) từng luồng')
    ax.set_ylabel('Mbps')
    ax.grid(True, axis='y')

    # 3. Jitter
    ax = axs[2]
    ax.bar(labels, jitters, color='green')
    ax.set_title('Jitter (ms) từng luồng')
    ax.set_ylabel('ms')
    ax.grid(True, axis='y')

    # 4. Lost/Total
    ax = axs[3]
    ax.bar(labels, loss_rates, color='red')
    ax.set_title('Tỷ lệ mất gói (%) từng luồng')
    ax.set_ylabel('Tỷ lệ mất gói (%)')
    ax.grid(True, axis='y')

    # 5. Biểu đồ Ping (nếu có)
    if ping_times:
        ax = axs[4]
        ax.plot(ping_times, marker='.', linestyle='-', color='purple')
        ax.set_title('Phân bố độ trễ Ping')
        ax.set_xlabel('Chỉ số gói')
        ax.set_ylabel('Độ trễ (ms)')
        ax.grid(True)

    # Thông số tổng hợp dưới biểu đồ
    avg_jitter = np.mean([j for j in jitters if j > 0]) if any(jitters) else None
    avg_loss = np.mean([r for r in loss_rates if r > 0]) if any(loss_rates) else None
    ping_avg = ping_min = ping_max = ping_jitter = None
    if ping_times:
        ping_avg = np.mean(ping_times)
        ping_min = np.min(ping_times)
        ping_max = np.max(ping_times)
        ping_jitter = np.std(ping_times)

    stats_text = (
        f"Tổng dữ liệu truyền: {sum(transfers):.1f} MB\n"
        f"Tốc độ tổng trung bình: {sum(bitrates):.1f} Mbps\n"
        + (f"Jitter trung bình: {avg_jitter:.3f} ms\n" if avg_jitter is not None else "")
        + (f"Tỷ lệ mất gói trung bình: {avg_loss:.3f} %\n" if avg_loss is not None else "")
        + (f"Ping trung bình: {ping_avg:.2f} ms, min: {ping_min:.2f} ms, max: {ping_max:.2f} ms, jitter: {ping_jitter:.2f} ms" if ping_avg is not None else "")
    )

    fig.suptitle("Phân tích Hiệu Năng Thuật Toán RED (UDP)", fontsize=18, fontweight='bold')
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12,
                bbox=dict(facecolor='lightgray', alpha=0.5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_img, dpi=300)
    plt.close(fig)

def plot_red_stats_analysis(intervals, out_img):
    if not intervals:
        print("Không có dữ liệu thống kê RED để vẽ biểu đồ.")
        return
    
    times = [interval['time'] for interval in intervals]
    bytes_sent = [interval['bytes_sent'] / (1024*1024) for interval in intervals]  # Convert to MB
    pkts_sent = [interval['pkts_sent'] / 1000 for interval in intervals]  # Convert to thousands
    dropped = [interval['dropped'] for interval in intervals]
    marked = [interval['marked'] for interval in intervals]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Bytes sent over time
    ax = axs[0, 0]
    ax.plot(range(len(times)), bytes_sent, marker='o', linewidth=2, markersize=8)
    ax.set_title('Bytes Sent theo thời gian')
    ax.set_xlabel('Mốc thời gian')
    ax.set_ylabel('Bytes (MB)')
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45)
    ax.grid(True)

    # 2. Packets sent over time
    ax = axs[0, 1]
    ax.plot(range(len(times)), pkts_sent, marker='s', linewidth=2, markersize=8, color='green')
    ax.set_title('Packets Sent theo thời gian')
    ax.set_xlabel('Mốc thời gian')
    ax.set_ylabel('Packets (thousands)')
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45)
    ax.grid(True)

    # 3. Dropped packets over time
    ax = axs[1, 0]
    ax.bar(range(len(times)), dropped, color='red', alpha=0.7)
    ax.set_title('Packets Dropped theo thời gian')
    ax.set_xlabel('Mốc thời gian')
    ax.set_ylabel('Số gói bị drop')
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45)
    ax.grid(True, axis='y')

    # 4. Marked packets over time
    ax = axs[1, 1]
    ax.bar(range(len(times)), marked, color='orange', alpha=0.7)
    ax.set_title('Packets Marked theo thời gian')
    ax.set_xlabel('Mốc thời gian')
    ax.set_ylabel('Số gói bị mark')
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45)
    ax.grid(True, axis='y')

    fig.suptitle("Phân tích Thống Kê RED theo Thời gian", fontsize=18, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- Main ---
if __name__ == '__main__':
    iperf_file_udp = r'd:\Ths\20252\v2\red\iperf_red_udp.log'
    iperf_file_tcp = r'd:\Ths\20252\v2\red\iperf_red.log'
    red_stats_file = r'd:\Ths\20252\v2\red\red_stats.txt'
    red_stats_interval_file = r'd:\Ths\20252\v2\red\red_stats_interval.txt'
    ping_file = r'd:\Ths\20252\v2\red\ping_red.log'
    target_bandwidth = 100  # Mbps (ví dụ)

    print("\nPhân tích file iperf_red_udp.log (RED UDP)...")
    data_udp, final_stats_udp = parse_iperf_log_udp(iperf_file_udp)
    print("\nPhân tích file iperf_red.log (RED TCP)...")
    data_tcp, final_stats_tcp = parse_iperf_log_tcp(iperf_file_tcp)
    print("\nPhân tích file red_stats.txt ...")
    sent_pkt, dropped, marked = parse_red_stats(red_stats_file)
    print("\nPhân tích file red_stats_interval.txt ...")
    intervals = parse_red_stats_interval(red_stats_interval_file)
    print("\nPhân tích file ping_red.log ...")
    ping_times = parse_ping_log(ping_file)

    print("\n--- Kết quả phân tích RED UDP ---")
    analyze_bandwidth(final_stats_udp, target_bandwidth)
    analyze_packet_loss(sent_pkt, dropped, marked)
    analyze_latency(ping_times)
    analyze_fairness(final_stats_udp, ping_times)
    plot_red_analysis_udp(final_stats_udp, ping_times, 'red_performance_analysis_udp.png')

    print("\n--- Kết quả phân tích RED TCP ---")
    analyze_bandwidth(final_stats_tcp, target_bandwidth)
    analyze_packet_loss(sent_pkt, dropped, marked)
    analyze_latency(ping_times)
    analyze_fairness(final_stats_tcp, ping_times)
    plot_red_analysis_tcp(data_tcp, final_stats_tcp, ping_times, 'red_performance_analysis_tcp.png')

    print("\n--- Phân tích thống kê RED ---")
    plot_red_stats_analysis(intervals, 'red_stats_analysis.png')

    # Xuất báo cáo thống kê ra file CSV cho UDP
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    report_udp = {
        'Stream': ['6', '8', '10', '12', 'Tổng'],
        'Dữ liệu (MB)': [final_stats_udp['stream6']['transfer'] if 'stream6' in final_stats_udp else 0,
                        final_stats_udp['stream8']['transfer'] if 'stream8' in final_stats_udp else 0,
                        final_stats_udp['stream10']['transfer'] if 'stream10' in final_stats_udp else 0,
                        final_stats_udp['stream12']['transfer'] if 'stream12' in final_stats_udp else 0,
                        sum(final_stats_udp[s]['transfer'] if s in final_stats_udp else 0 for s in streams)],
        'Bitrate (Mbps)': [final_stats_udp['stream6']['bitrate'] if 'stream6' in final_stats_udp else 0,
                          final_stats_udp['stream8']['bitrate'] if 'stream8' in final_stats_udp else 0,
                          final_stats_udp['stream10']['bitrate'] if 'stream10' in final_stats_udp else 0,
                          final_stats_udp['stream12']['bitrate'] if 'stream12' in final_stats_udp else 0,
                          sum(final_stats_udp[s]['bitrate'] if s in final_stats_udp else 0 for s in streams)],
        'Jitter (ms)': [final_stats_udp['stream6']['jitter'] if 'stream6' in final_stats_udp else 0,
                       final_stats_udp['stream8']['jitter'] if 'stream8' in final_stats_udp else 0,
                       final_stats_udp['stream10']['jitter'] if 'stream10' in final_stats_udp else 0,
                       final_stats_udp['stream12']['jitter'] if 'stream12' in final_stats_udp else 0,
                       np.mean([final_stats_udp[s]['jitter'] for s in streams if s in final_stats_udp])]
    }
    df_udp = pd.DataFrame(report_udp)
    print("\nBáo Cáo Thống Kê Hiệu Năng RED (UDP):")
    print(df_udp.to_string(index=False))
    out_csv_udp = 'red_performance_report_udp.csv'
    df_udp.to_csv(out_csv_udp, index=False)
    print(f"\nĐã lưu báo cáo UDP ra file: {out_csv_udp}")

    # Xuất báo cáo thống kê ra file CSV cho TCP
    report_tcp = {
        'Stream': ['6', '8', '10', '12', 'Tổng'],
        'Dữ liệu (MB)': [final_stats_tcp['stream6']['transfer'] if 'stream6' in final_stats_tcp else 0,
                        final_stats_tcp['stream8']['transfer'] if 'stream8' in final_stats_tcp else 0,
                        final_stats_tcp['stream10']['transfer'] if 'stream10' in final_stats_tcp else 0,
                        final_stats_tcp['stream12']['transfer'] if 'stream12' in final_stats_tcp else 0,
                        sum(final_stats_tcp[s]['transfer'] if s in final_stats_tcp else 0 for s in streams)],
        'Bitrate (Mbps)': [final_stats_tcp['stream6']['bitrate'] if 'stream6' in final_stats_tcp else 0,
                          final_stats_tcp['stream8']['bitrate'] if 'stream8' in final_stats_tcp else 0,
                          final_stats_tcp['stream10']['bitrate'] if 'stream10' in final_stats_tcp else 0,
                          final_stats_tcp['stream12']['bitrate'] if 'stream12' in final_stats_tcp else 0,
                          sum(final_stats_tcp[s]['bitrate'] if s in final_stats_tcp else 0 for s in streams)],
        'Truyền lại': [
            final_stats_tcp['stream6'].get('total_retr', 0) if 'stream6' in final_stats_tcp else 0,
            final_stats_tcp['stream8'].get('total_retr', 0) if 'stream8' in final_stats_tcp else 0,
            final_stats_tcp['stream10'].get('total_retr', 0) if 'stream10' in final_stats_tcp else 0,
            final_stats_tcp['stream12'].get('total_retr', 0) if 'stream12' in final_stats_tcp else 0,
            final_stats_tcp.get('total_retr', 0)
        ]
    }
    df_tcp = pd.DataFrame(report_tcp)
    print("\nBáo Cáo Thống Kê Hiệu Năng RED (TCP):")
    print(df_tcp.to_string(index=False))
    out_csv_tcp = 'red_performance_report_tcp.csv'
    df_tcp.to_csv(out_csv_tcp, index=False)
    print(f"\nĐã lưu báo cáo TCP ra file: {out_csv_tcp}")

    # Xuất báo cáo thống kê RED
    if intervals:
        report_red = {
            'Thời gian': [interval['time'] for interval in intervals],
            'Bytes Sent (MB)': [interval['bytes_sent'] / (1024*1024) for interval in intervals],
            'Packets Sent': [interval['pkts_sent'] for interval in intervals],
            'Dropped': [interval['dropped'] for interval in intervals],
            'Marked': [interval['marked'] for interval in intervals]
        }
        df_red = pd.DataFrame(report_red)
        print("\nBáo Cáo Thống Kê RED theo thời gian:")
        print(df_red.to_string(index=False))
        out_csv_red = 'red_stats_report.csv'
        df_red.to_csv(out_csv_red, index=False)
        print(f"\nĐã lưu báo cáo RED ra file: {out_csv_red}") 