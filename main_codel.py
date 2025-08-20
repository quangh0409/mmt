import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Phân tích iperf_codel_tcp.log (TCP) ---
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

    # Interval line regex (with Retr and Cwnd)
    interval_pattern = re.compile(
        r'\[\s*(\d+|SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)\s+([\d\.]+)\s+KBytes'
    )
    
    # Summary line regex
    summary_pattern = re.compile(
        r'\[\s*(\d+|SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)'
    )

    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse interval lines
            match = interval_pattern.match(line)
            if match:
                stream_id = match.group(1)
                start_time = float(match.group(2))
                end_time = float(match.group(3))
                transfer = float(match.group(4))
                bitrate = float(match.group(5))
                retr = int(match.group(6))
                cwnd = float(match.group(7))
                
                # Calculate average time for this interval
                avg_time = (start_time + end_time) / 2
                
                # Store data based on stream ID
                if stream_id == '6':
                    data['time'].append(avg_time)
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
                elif stream_id == 'SUM':
                    data['sum_bitrate'].append(bitrate)
                    data['sum_retr'].append(retr)
            
            # Parse summary lines
            summary_match = summary_pattern.match(line)
            if summary_match and 'sender' in line:
                stream_id = summary_match.group(1)
                transfer = float(summary_match.group(4))
                bitrate = float(summary_match.group(5))
                retr = int(summary_match.group(6))
                
                if stream_id == 'SUM':
                    final_stats['total_transfer'] = transfer
                    final_stats['total_bitrate'] = bitrate
                    final_stats['total_retr'] = retr
                else:
                    final_stats[f'stream{stream_id}'] = {
                        'transfer': transfer,
                        'bitrate': bitrate,
                        'retr': retr
                    }

    # Calculate sum bitrate if not available
    if not data['sum_bitrate'] and data['stream6_bitrate']:
        data['sum_bitrate'] = [sum([
            data['stream6_bitrate'][i] if i < len(data['stream6_bitrate']) else 0,
            data['stream8_bitrate'][i] if i < len(data['stream8_bitrate']) else 0,
            data['stream10_bitrate'][i] if i < len(data['stream10_bitrate']) else 0,
            data['stream12_bitrate'][i] if i < len(data['stream12_bitrate']) else 0
        ]) for i in range(max(len(data['stream6_bitrate']), len(data['stream8_bitrate']), 
                              len(data['stream10_bitrate']), len(data['stream12_bitrate'])))]

    return data, final_stats

# --- Phân tích iperf_codel_udp.log (UDP) ---
def parse_iperf_log_udp(filename):
    data = {
        'time': [],
        'stream6_transfer': [],
        'stream8_transfer': [],
        'stream10_transfer': [],
        'stream12_transfer': [],
        'sum_transfer': [],
        'stream6_bitrate': [],
        'stream8_bitrate': [],
        'stream10_bitrate': [],
        'stream12_bitrate': [],
        'sum_bitrate': [],
        'stream6_datagrams': [],
        'stream8_datagrams': [],
        'stream10_datagrams': [],
        'stream12_datagrams': [],
        'sum_datagrams': [],
    }
    final_stats = {}
    
    # Interval line regex for UDP
    interval_pattern = re.compile(
        r'\[\s*(\d+|SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+KBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)'
    )
    
    # Summary line regex for UDP
    summary_pattern = re.compile(
        r'\[\s*(\d+|SUM)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+([\d\.]+)\s+ms\s+(\d+)/(\d+)\s+\(([\d\.]+)%\)'
    )

    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse interval lines
            interval_match = interval_pattern.match(line)
            if interval_match:
                stream_id = interval_match.group(1)
                start_time = float(interval_match.group(2))
                end_time = float(interval_match.group(3))
                transfer = float(interval_match.group(4)) / 1024  # Convert to MB
                bitrate = float(interval_match.group(5))
                datagrams = int(interval_match.group(6))
                
                # Calculate average time for this interval
                avg_time = (start_time + end_time) / 2
                
                # Store data based on stream ID
                if stream_id == '6':
                    data['time'].append(avg_time)
                    data['stream6_transfer'].append(transfer)
                    data['stream6_bitrate'].append(bitrate)
                    data['stream6_datagrams'].append(datagrams)
                elif stream_id == '8':
                    data['stream8_transfer'].append(transfer)
                    data['stream8_bitrate'].append(bitrate)
                    data['stream8_datagrams'].append(datagrams)
                elif stream_id == '10':
                    data['stream10_transfer'].append(transfer)
                    data['stream10_bitrate'].append(bitrate)
                    data['stream10_datagrams'].append(datagrams)
                elif stream_id == '12':
                    data['stream12_transfer'].append(transfer)
                    data['stream12_bitrate'].append(bitrate)
                    data['stream12_datagrams'].append(datagrams)
                elif stream_id == 'SUM':
                    data['sum_transfer'].append(transfer)
                    data['sum_bitrate'].append(bitrate)
                    data['sum_datagrams'].append(datagrams)
            
            # Parse summary lines
            summary_match = summary_pattern.match(line)
            if summary_match and 'receiver' in line:
                stream_id = summary_match.group(1)
                transfer = float(summary_match.group(4))
                bitrate = float(summary_match.group(5))
                jitter = float(summary_match.group(6))
                lost = int(summary_match.group(7))
                total = int(summary_match.group(8))
                loss_percent = float(summary_match.group(9))
                
                if stream_id == 'SUM':
                    final_stats['total_transfer'] = transfer
                    final_stats['total_bitrate'] = bitrate
                    final_stats['total_jitter'] = jitter
                    final_stats['total_lost'] = lost
                    final_stats['total_total'] = total
                    final_stats['total_loss_percent'] = loss_percent
                else:
                    final_stats[f'stream{stream_id}'] = {
                        'transfer': transfer,
                        'bitrate': bitrate,
                        'jitter': jitter,
                        'lost': lost,
                        'total': total,
                        'loss_percent': loss_percent
                    }

    # Calculate sum data if not available
    if not data['sum_bitrate'] and data['stream6_bitrate']:
        min_length = min(len(data['stream6_bitrate']), len(data['stream8_bitrate']), 
                        len(data['stream10_bitrate']), len(data['stream12_bitrate']))
        data['sum_bitrate'] = [sum([
            data['stream6_bitrate'][i] if i < len(data['stream6_bitrate']) else 0,
            data['stream8_bitrate'][i] if i < len(data['stream8_bitrate']) else 0,
            data['stream10_bitrate'][i] if i < len(data['stream10_bitrate']) else 0,
            data['stream12_bitrate'][i] if i < len(data['stream12_bitrate']) else 0
        ]) for i in range(min_length)]

    return data, final_stats

# --- Phân tích ping_codel.log ---
def parse_ping_log(filename):
    times = []
    
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse ping time
            match = re.search(r'time=([\d\.]+)\s+ms', line)
            if match:
                time = float(match.group(1))
                times.append(time)
    
    return times

# --- Phân tích CoDel stats (giả lập) ---
def parse_codel_stats(filename=None):
    # CoDel không có file stats riêng, trả về dữ liệu mặc định
    intervals = []
    return intervals

# --- Phân tích hiệu năng băng thông ---
def analyze_bandwidth(final_stats, target_bandwidth):
    if 'total_bitrate' in final_stats:
        achieved = final_stats['total_bitrate']
        efficiency = (achieved / target_bandwidth) * 100
        print(f"Băng thông đạt được: {achieved:.2f} Mbps")
        print(f"Hiệu suất: {efficiency:.2f}% so với mục tiêu {target_bandwidth} Mbps")
        return achieved, efficiency
    else:
        print("Không có dữ liệu băng thông.")
        return None, None

# --- Phân tích packet loss ---
def analyze_packet_loss(sent_pkt, dropped):
    if isinstance(sent_pkt, dict) and 'total_lost' in sent_pkt and 'total_total' in sent_pkt:
        # Trường hợp UDP với dict
        lost = sent_pkt['total_lost']
        total = sent_pkt['total_total']
        loss_rate = (lost / total) * 100 if total > 0 else 0
        print(f"Tổng gói gửi: {total}, Gói bị mất: {lost}")
        print(f"Tỷ lệ mất gói: {loss_rate:.2f}%")
        return loss_rate
    elif isinstance(sent_pkt, (int, float)) and sent_pkt > 0:
        # Trường hợp stats file
        loss_rate = (dropped / sent_pkt) * 100
        print(f"Tổng gói gửi: {sent_pkt}, Gói bị loại: {dropped}")
        print(f"Tỷ lệ mất gói: {loss_rate:.2f}%")
        return loss_rate
    else:
        print("Không có dữ liệu packet loss.")
        return None

# --- Phân tích độ trễ ---
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

# --- Phân tích công bằng ---
def analyze_fairness(final_stats, ping_times=None):
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    
    # Tính các thông số tổng hợp
    fairness = 0
    avg_bitrates = [final_stats[s]['bitrate'] if s in final_stats and 'bitrate' in final_stats[s] else 0 for s in streams]
    
    if any(avg_bitrates):
        fairness = (sum(avg_bitrates) ** 2) / (len(avg_bitrates) * sum(b ** 2 for b in avg_bitrates)) if max(avg_bitrates) > 0 else 0

    total_transfer = sum(final_stats[s]['transfer'] if s in final_stats and 'transfer' in final_stats[s] else 0 for s in streams)
    total_bitrate = sum(avg_bitrates)
    total_retr = final_stats.get('total_retr', 0)
    
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
        'ping_avg': ping_avg,
        'ping_min': ping_min,
        'ping_max': ping_max,
        'ping_jitter': ping_jitter,
    }

# --- Vẽ biểu đồ phân tích TCP CoDel ---
def plot_codel_analysis_tcp(data, final_stats, ping_times, out_img):
    n_plots = 6 if ping_times else 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    fig.suptitle('Phân tích hiệu năng CoDel TCP', fontsize=16, fontweight='bold')
    
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]

    # Đảm bảo tất cả các stream có cùng độ dài
    min_length = min(
        len(data.get('time', [])),
        *(len(data.get(f'stream{i}_bitrate', [])) for i in [6, 8, 10, 12]),
        len(data.get('sum_bitrate', []))
    ) if data.get('time') else 0
    
    time_data = data['time'][:min_length]
    stream_bitrates = [data.get(f'stream{i}_bitrate', [])[:min_length] for i in [6, 8, 10, 12]]
    sum_bitrate = data.get('sum_bitrate', [])[:min_length]
    stream_retrs = [data.get(f'stream{i}_retr', [])[:min_length] for i in [6, 8, 10, 12]]
    sum_retr = data.get('sum_retr', [])[:min_length]
    stream_cwnds = [data.get(f'stream{i}_cwnd', [])[:min_length] for i in [6, 8, 10, 12]]

    avg_bitrates = [final_stats[s]['bitrate'] if s in final_stats and 'bitrate' in final_stats[s] else 0 for s in streams]
    total_retr = final_stats.get('total_retr', 0)
    retr_vals = [final_stats[s]['retr'] if s in final_stats and 'retr' in final_stats[s] else 0 for s in streams]
    
    # 1. Bitrate theo thời gian
    ax = axs[0]
    if time_data:
        plotted_labels = set()
        for i, s_bitrate in enumerate(stream_bitrates):
            if len(s_bitrate) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, s_bitrate, label=labels[i], linewidth=2)
                plotted_labels.add(labels[i])
        if len(sum_bitrate) > 0 and 'Tổng' not in plotted_labels:
            ax.plot(time_data, sum_bitrate, 'k--', label='Tổng', linewidth=3)
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Bitrate (Mbps)')
        ax.set_title('Bitrate theo thời gian (CoDel TCP)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Retransmissions theo thời gian
    ax = axs[1]
    if time_data:
        plotted_labels = set()
        for i, s_retr in enumerate(stream_retrs):
            if len(s_retr) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, s_retr, label=labels[i], linewidth=2)
                plotted_labels.add(labels[i])
        if len(sum_retr) > 0 and 'Tổng' not in plotted_labels:
            ax.plot(time_data, sum_retr, 'k--', label='Tổng', linewidth=3)
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Retransmissions')
        ax.set_title('Retransmissions theo thời gian')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Tốc độ trung bình từng luồng
    ax = axs[2]
    if any(avg_bitrates):
        bars = ax.bar(labels, avg_bitrates, color=['blue', 'red', 'green', 'magenta'], alpha=0.7)
        ax.set_title('Tốc độ trung bình từng luồng')
        ax.set_ylabel('Bitrate (Mbps)')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, value in zip(bars, avg_bitrates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{value:.1f}', ha='center', va='bottom')
    
    # 4. Tổng số gói truyền lại từng luồng và tổng
    ax = axs[3]
    bar_labels = labels + ['Tổng']
    bar_vals = retr_vals + [total_retr]
    if any(bar_vals):
        bars = ax.bar(bar_labels, bar_vals, color=['blue', 'red', 'green', 'magenta', 'black'], alpha=0.7)
        ax.set_title('Tổng số gói truyền lại từng luồng và tổng')
        ax.set_ylabel('Retransmissions')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, value in zip(bars, bar_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{value}', ha='center', va='bottom')

    # 5. Congestion Window theo thời gian
    ax = axs[4]
    if time_data:
        plotted_labels = set()
        for i, s_cwnd in enumerate(stream_cwnds):
            if len(s_cwnd) > 0 and labels[i] not in plotted_labels:
                ax.plot(time_data, s_cwnd, label=labels[i], linewidth=2)
                plotted_labels.add(labels[i])
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Congestion Window (KB)')
        ax.set_title('Congestion Window theo thời gian')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Ping latency theo thời gian (nếu có)
    if ping_times:
        ax = axs[5]
        ping_indices = list(range(1, len(ping_times) + 1))
        ax.plot(ping_indices, ping_times, 'g-', linewidth=2, alpha=0.7)
        ax.axhline(y=np.mean(ping_times), color='r', linestyle='--', label=f'Trung bình: {np.mean(ping_times):.2f} ms')
        ax.set_xlabel('Ping sequence')
        ax.set_ylabel('Thời gian (ms)')
        ax.set_title('Ping latency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Thông số tổng hợp dưới biểu đồ (sử dụng figtext)
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
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for figtext
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig) # Use plt.close(fig) to prevent displaying multiple figures if running in a loop and for memory management
    print(f"Đã lưu biểu đồ TCP CoDel: {out_img}")

# --- Vẽ biểu đồ phân tích UDP CoDel ---
def plot_codel_analysis_udp(data, final_stats, ping_times, out_img):
    n_plots = 5 if ping_times else 4
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    fig.suptitle('Phân tích hiệu năng CoDel UDP', fontsize=16, fontweight='bold')
    
    streams = ['stream6', 'stream8', 'stream10', 'stream12']
    labels = [f'Luồng {s[-2:]}' for s in streams]
    transfers = [final_stats[s]['transfer'] if s in final_stats and 'transfer' in final_stats[s] else 0 for s in streams]
    bitrates = [final_stats[s]['bitrate'] if s in final_stats and 'bitrate' in final_stats[s] else 0 for s in streams]
    jitters = [final_stats[s]['jitter'] if s in final_stats and 'jitter' in final_stats[s] else 0 for s in streams]
    losts = [final_stats[s]['lost'] if s in final_stats and 'lost' in final_stats[s] else 0 for s in streams]
    totals = [final_stats[s]['total'] if s in final_stats and 'total' in final_stats[s] else 1 for s in streams]
    loss_percents = [100 * lost / total if total > 0 else 0 for lost, total in zip(losts, totals)]

    # 1. Transfer theo stream
    ax = axs[0]
    if any(transfers):
        bars = ax.bar(labels, transfers, color=['blue', 'red', 'green', 'magenta'], alpha=0.7)
        ax.set_title('Transfer (MB) từng luồng (CoDel UDP)')
        ax.set_ylabel('MB')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, value in zip(bars, transfers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{value:.1f}', ha='center', va='bottom')
    
    # 2. Bitrate theo stream
    ax = axs[1]
    if any(bitrates):
        bars = ax.bar(labels, bitrates, color=['blue', 'red', 'green', 'magenta'], alpha=0.7)
        ax.set_title('Bitrate (Mbps) từng luồng')
        ax.set_ylabel('Mbps')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, value in zip(bars, bitrates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{value:.2f}', ha='center', va='bottom')
    
    # 3. Jitter theo stream
    ax = axs[2]
    if any(jitters):
        bars = ax.bar(labels, jitters, color=['blue', 'red', 'green', 'magenta'], alpha=0.7)
        ax.set_title('Jitter (ms) từng luồng')
        ax.set_ylabel('ms')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, value in zip(bars, jitters):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Packet loss theo stream
    ax = axs[3]
    if any(loss_percents):
        bars = ax.bar(labels, loss_percents, color=['blue', 'red', 'green', 'magenta'], alpha=0.7)
        ax.set_title('Tỷ lệ mất gói (%) từng luồng')
        ax.set_ylabel('Packet Loss (%)')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, value in zip(bars, loss_percents):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.1f}%', ha='center', va='bottom')
    
    # 5. Ping latency (nếu có)
    if ping_times:
        ax = axs[4]
        ping_indices = list(range(1, len(ping_times) + 1))
        ax.plot(ping_indices, ping_times, 'g-', linewidth=2, alpha=0.7)
        ax.axhline(y=np.mean(ping_times), color='r', linestyle='--', label=f'Trung bình: {np.mean(ping_times):.2f} ms')
        ax.set_xlabel('Ping sequence')
        ax.set_ylabel('Thời gian (ms)')
        ax.set_title('Ping latency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Tổng kết thống kê dưới biểu đồ (sử dụng figtext)
    total_transfer = final_stats.get('total_transfer', 0)
    total_bitrate = final_stats.get('total_bitrate', 0)
    total_jitter = final_stats.get('total_jitter', 0)
    total_lost = final_stats.get('total_lost', 0)
    total_total = final_stats.get('total_total', 0)
    total_loss_percent = final_stats.get('total_loss_percent', 0)
    
    # Tính fairness
    fairness = 0
    if any(bitrates):
        fairness = (sum(bitrates) ** 2) / (len(bitrates) * sum(b ** 2 for b in bitrates)) if max(bitrates) > 0 else 0
    
    ping_avg = ping_min = ping_max = ping_jitter = None
    if ping_times:
        ping_avg = np.mean(ping_times)
        ping_min = np.min(ping_times)
        ping_max = np.max(ping_times)
        ping_jitter = np.std(ping_times)
    
    stats_text = (
        f"Tổng dữ liệu truyền: {total_transfer:.1f} MB\n"
        f"Tốc độ tổng: {total_bitrate:.2f} Mbps\n"
        f"Jitter tổng: {total_jitter:.3f} ms\n"
        f"Packet loss: {total_lost}/{total_total} ({total_loss_percent:.1f}%)\n"
        f"Tỷ lệ công bằng (Jain): {fairness:.3f}\n"
        + (f"Ping trung bình: {ping_avg:.2f} ms, jitter: {ping_jitter:.2f} ms" if ping_avg is not None else "")
    )
    
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=11,
              verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for figtext
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig) # Use plt.close(fig) to prevent displaying multiple figures if running in a loop and for memory management
    print(f"Đã lưu biểu đồ UDP CoDel: {out_img}")

# --- Xuất báo cáo CSV ---
def export_codel_report(final_stats_tcp, final_stats_udp, ping_times, out_csv):
    report_data = []
    
    # Thống kê TCP
    if final_stats_tcp:
        streams = ['stream6', 'stream8', 'stream10', 'stream12']
        for stream in streams:
            if stream in final_stats_tcp:
                row = {
                    'Protocol': 'TCP',
                    'Stream': stream,
                    'Transfer_MB': final_stats_tcp[stream].get('transfer', 0),
                    'Bitrate_Mbps': final_stats_tcp[stream].get('bitrate', 0),
                    'Retransmissions': final_stats_tcp[stream].get('retr', 0),
                    'Jitter_ms': None,
                    'Packet_Loss_%': None
                }
                report_data.append(row)
        
        # Tổng TCP
        if 'total_transfer' in final_stats_tcp:
            row = {
                'Protocol': 'TCP',
                'Stream': 'TOTAL',
                'Transfer_MB': final_stats_tcp['total_transfer'],
                'Bitrate_Mbps': final_stats_tcp['total_bitrate'],
                'Retransmissions': final_stats_tcp['total_retr'],
                'Jitter_ms': None,
                'Packet_Loss_%': None
            }
            report_data.append(row)
    
    # Thống kê UDP
    if final_stats_udp:
        streams = ['stream6', 'stream8', 'stream10', 'stream12']
        for stream in streams:
            if stream in final_stats_udp:
                row = {
                    'Protocol': 'UDP',
                    'Stream': stream,
                    'Transfer_MB': final_stats_udp[stream].get('transfer', 0),
                    'Bitrate_Mbps': final_stats_udp[stream].get('bitrate', 0),
                    'Retransmissions': None,
                    'Jitter_ms': final_stats_udp[stream].get('jitter', 0),
                    'Packet_Loss_%': final_stats_udp[stream].get('loss_percent', 0)
                }
                report_data.append(row)
        
        # Tổng UDP
        if 'total_transfer' in final_stats_udp:
            row = {
                'Protocol': 'UDP',
                'Stream': 'TOTAL',
                'Transfer_MB': final_stats_udp['total_transfer'],
                'Bitrate_Mbps': final_stats_udp['total_bitrate'],
                'Retransmissions': None,
                'Jitter_ms': final_stats_udp['total_jitter'],
                'Packet_Loss_%': final_stats_udp['total_loss_percent']
            }
            report_data.append(row)
    
    # Thống kê ping
    if ping_times:
        ping_stats = {
            'Protocol': 'PING',
            'Stream': 'LATENCY',
            'Transfer_MB': None,
            'Bitrate_Mbps': None,
            'Retransmissions': None,
            'Jitter_ms': np.std(ping_times),
            'Packet_Loss_%': None
        }
        report_data.append(ping_stats)
        
        ping_avg = {
            'Protocol': 'PING',
            'Stream': 'AVERAGE',
            'Transfer_MB': None,
            'Bitrate_Mbps': None,
            'Retransmissions': None,
            'Jitter_ms': np.mean(ping_times),
            'Packet_Loss_%': None
        }
        report_data.append(ping_avg)
    
    # Tạo DataFrame và xuất CSV
    df = pd.DataFrame(report_data)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Đã xuất báo cáo CSV: {out_csv}")

# --- Hàm chính ---
if __name__ == '__main__':
    # Đường dẫn file
    iperf_file_udp = r'd:\Ths\20252\v2\codel\iperf_codel_udp.log'
    iperf_file_tcp = r'd:\Ths\20252\v2\codel\iperf_codel_tcp.log'
    ping_file = r'd:\Ths\20252\v2\codel\ping_codel.log'
    target_bandwidth = 100  # Mbps (ví dụ)
    
    print("="*80)
    print("PHÂN TÍCH HIỆU NĂNG THUẬT TOÁN CODEL")
    print("="*80)
    
    print("\nPhân tích file iperf_codel_udp.log (UDP)...")
    data_udp, final_stats_udp = parse_iperf_log_udp(iperf_file_udp)
    
    print("\nPhân tích file iperf_codel_tcp.log (TCP)...")
    data_tcp, final_stats_tcp = parse_iperf_log(iperf_file_tcp)
    
    print("\nPhân tích file ping_codel.log ...")
    ping_times = parse_ping_log(ping_file)
    
    # Phân tích CoDel stats (không có file thực tế)
    codel_intervals = parse_codel_stats()
    
    print("\n--- Kết quả phân tích UDP ---")
    analyze_bandwidth(final_stats_udp, target_bandwidth)
    analyze_packet_loss(final_stats_udp, 0)  # Sử dụng final_stats_udp làm sent_pkt
    analyze_latency(ping_times)
    fairness_udp = analyze_fairness(final_stats_udp, ping_times)
    plot_codel_analysis_udp(data_udp, final_stats_udp, ping_times, 'codel_performance_analysis_udp.png')
    
    print("\n--- Kết quả phân tích TCP ---")
    analyze_bandwidth(final_stats_tcp, target_bandwidth)
    analyze_packet_loss(0, 0)  # CoDel không có file stats riêng
    analyze_latency(ping_times)
    fairness_tcp = analyze_fairness(final_stats_tcp, ping_times)
    plot_codel_analysis_tcp(data_tcp, final_stats_tcp, ping_times, 'codel_performance_analysis_tcp.png')
    
    # Xuất báo cáo CSV
    export_codel_report(final_stats_tcp, final_stats_udp, ping_times, 'codel_performance_report.csv')
    
    # So sánh với các thuật toán khác
    print("\n" + "="*80)
    print("SO SÁNH HIỆU NĂNG CÁC THUẬT TOÁN QUẢN LÝ HÀNG ĐỢI")
    print("="*80)
    print(f"CoDel - Tổng bitrate TCP: {final_stats_tcp.get('total_bitrate', 0):.1f} Mbps")
    print(f"CoDel - Tổng retrans: {final_stats_tcp.get('total_retr', 0)}")
    print(f"CoDel - Tổng bitrate UDP: {final_stats_udp.get('total_bitrate', 0):.1f} Mbps")
    if 'total_lost' in final_stats_udp and 'total_total' in final_stats_udp:
        total_lost = final_stats_udp['total_lost']
        total_total = final_stats_udp['total_total']
        if total_total > 0:
            print(f"CoDel - Packet loss UDP: {total_lost}/{total_total} ({total_lost/total_total*100:.2f}%)")
        else:
            print(f"CoDel - Packet loss UDP: {total_lost}/{total_total} (0%)")
    else:
        print("CoDel - Không có dữ liệu packet loss UDP")
    
    if ping_times:
        print(f"CoDel - Ping trung bình: {np.mean(ping_times):.2f} ms, jitter: {np.std(ping_times):.2f} ms")
    
    print("\n" + "="*80)
    print("PHÂN TÍCH HOÀN THÀNH!")
    print("="*80) 