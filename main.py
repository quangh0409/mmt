import re
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# Hàm phân tích file log
def parse_pfifo_log(filename):
    interval_pattern = re.compile(
        r'\[\s*(\d+)\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)\s+[\d\.]+\s+KBytes'
    )
    sum_pattern = re.compile(
        r'\[SUM\]\s+([\d\.]+)-([\d\.]+)\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)'
    )
    final_pattern = re.compile(
        r'\[\s*(\d+)\]\s+0\.00-60\.\d+\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)\s+sender'
    )
    final_sum_pattern = re.compile(
        r'\[SUM\]\s+0\.00-60\.\d+\s+sec\s+([\d\.]+)\s+MBytes\s+([\d\.]+)\s+Mbits/sec\s+(\d+)\s+sender'
    )

    data = {
        'time': [],
        'stream5_bitrate': [],
        'stream7_bitrate': [],
        'stream9_bitrate': [],
        'stream11_bitrate': [],
        'sum_bitrate': [],
        'stream5_retr': [],
        'stream7_retr': [],
        'stream9_retr': [],
        'stream11_retr': [],
        'sum_retr': []
    }
    final_stats = {}

    with open(filename, 'r') as f:
        for line in f:
            interval_match = interval_pattern.search(line)
            sum_match = sum_pattern.search(line)
            final_match = final_pattern.search(line)
            final_sum_match = final_sum_pattern.search(line)

            if interval_match:
                stream_id, start, end, transfer, bitrate, retr = interval_match.groups()
                if stream_id in ['5', '7', '9', '11']:
                    if float(start) not in data['time']:
                        data['time'].append(float(start))
                    if stream_id == '5':
                        data['stream5_bitrate'].append(float(bitrate))
                        data['stream5_retr'].append(int(retr))
                    elif stream_id == '7':
                        data['stream7_bitrate'].append(float(bitrate))
                        data['stream7_retr'].append(int(retr))
                    elif stream_id == '9':
                        data['stream9_bitrate'].append(float(bitrate))
                        data['stream9_retr'].append(int(retr))
                    elif stream_id == '11':
                        data['stream11_bitrate'].append(float(bitrate))
                        data['stream11_retr'].append(int(retr))
            elif sum_match:
                start, end, transfer, bitrate, retr = sum_match.groups()
                data['sum_bitrate'].append(float(bitrate))
                data['sum_retr'].append(int(retr))
            elif final_match:
                stream_id, transfer, bitrate, retr = final_match.groups()
                final_stats[f'stream{stream_id}'] = {
                    'transfer': float(transfer),
                    'bitrate': float(bitrate),
                    'total_retr': int(retr)
                }
            elif final_sum_match:
                transfer, bitrate, retr = final_sum_match.groups()
                final_stats['total_transfer'] = float(transfer)
                final_stats['total_bitrate'] = float(bitrate)
                final_stats['total_retr'] = int(retr)

    # Nếu thiếu tổng, tự tính lại
    if 'total_transfer' not in final_stats:
        final_stats['total_transfer'] = sum([final_stats[s]['transfer'] for s in final_stats if s.startswith('stream')])
    if 'total_bitrate' not in final_stats:
        final_stats['total_bitrate'] = sum([final_stats[s]['bitrate'] for s in final_stats if s.startswith('stream')])
    if 'total_retr' not in final_stats:
        final_stats['total_retr'] = sum([final_stats[s]['total_retr'] for s in final_stats if s.startswith('stream')])

    return data, final_stats

# Hàm vẽ biểu đồ
def plot_pfifo_analysis(data, final_stats, algo_name, out_img):
    min_len = min(len(data['time']), len(data['stream5_bitrate']), len(data['stream7_bitrate']),
                  len(data['stream9_bitrate']), len(data['stream11_bitrate']), len(data['sum_bitrate']))
    time_points = data['time'][:min_len]
    stream5_bitrate = data['stream5_bitrate'][:min_len]
    stream7_bitrate = data['stream7_bitrate'][:min_len]
    stream9_bitrate = data['stream9_bitrate'][:min_len]
    stream11_bitrate = data['stream11_bitrate'][:min_len]
    sum_bitrate = data['sum_bitrate'][:min_len]
    stream5_retr = data['stream5_retr'][:min_len]
    stream7_retr = data['stream7_retr'][:min_len]
    stream9_retr = data['stream9_retr'][:min_len]
    stream11_retr = data['stream11_retr'][:min_len]
    sum_retr = data['sum_retr'][:min_len]
    plt.figure(figsize=(16, 12))
    plt.suptitle(f'Phân Tích Hiệu Năng Thuật Toán {algo_name}', fontsize=18, fontweight='bold')

    gs = GridSpec(3, 2, figure=plt.gcf())

    ax1 = plt.subplot(gs[0, :])
    ax1.plot(time_points, stream5_bitrate, 'b-', label='Luồng 5', linewidth=1.5)
    ax1.plot(time_points, stream7_bitrate, 'g-', label='Luồng 7', linewidth=1.5)
    ax1.plot(time_points, stream9_bitrate, 'r-', label='Luồng 9', linewidth=1.5)
    ax1.plot(time_points, stream11_bitrate, 'c-', label='Luồng 11', linewidth=1.5)
    ax1.plot(time_points, sum_bitrate, 'm--', label='Tổng', linewidth=2.5)
    ax1.set_title('Tốc Độ Bitrate Theo Thời Gian', fontsize=14)
    ax1.set_xlabel('Thời Gian (giây)', fontsize=12)
    ax1.set_ylabel('Tốc Độ (Mbits/sec)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(sum_bitrate) * 1.2 if sum_bitrate else 1)

    ax2 = plt.subplot(gs[1, :])
    ax2.plot(time_points, stream5_retr, 'b-', label='Luồng 5', linewidth=1.5)
    ax2.plot(time_points, stream7_retr, 'g-', label='Luồng 7', linewidth=1.5)
    ax2.plot(time_points, stream9_retr, 'r-', label='Luồng 9', linewidth=1.5)
    ax2.plot(time_points, stream11_retr, 'c-', label='Luồng 11', linewidth=1.5)
    ax2.plot(time_points, sum_retr, 'm--', label='Tổng', linewidth=2.5)
    ax2.set_title('Số Gói Truyền Lại Theo Thời Gian', fontsize=14)
    ax2.set_xlabel('Thời Gian (giây)', fontsize=12)
    ax2.set_ylabel('Số Gói Truyền Lại', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')

    ax3 = plt.subplot(gs[2, 0])
    streams = ['stream5', 'stream7', 'stream9', 'stream11']
    avg_bitrates = [final_stats[s]['bitrate'] if s in final_stats else 0 for s in streams]
    colors = ['blue', 'green', 'red', 'cyan']
    bars = ax3.bar(streams, avg_bitrates, color=colors)
    ax3.set_title('Tốc Độ Trung Bình Các Luồng', fontsize=14)
    ax3.set_ylabel('Mbits/sec', fontsize=12)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    ax4 = plt.subplot(gs[2, 1])
    total_retr = [final_stats[s]['total_retr'] if s in final_stats else 0 for s in streams]
    total_retr.append(final_stats['total_retr'])
    labels = ['Luồng 5', 'Luồng 7', 'Luồng 9', 'Luồng 11', 'Tổng']
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    bars = ax4.bar(labels, total_retr, color=colors)
    ax4.set_title('Tổng Số Gói Truyền Lại', fontsize=14)
    ax4.set_ylabel('Số Gói', fontsize=12)
    ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')

    fairness = "N/A"
    if max(avg_bitrates) > 0:
        fairness = f"{min(avg_bitrates)/max(avg_bitrates)*100:.1f}%"

    stats_text = (
        f"Tổng dữ liệu truyền: {sum(final_stats[s]['transfer'] if s in final_stats else 0 for s in streams):.1f} MB\n"
        f"Tốc độ tổng trung bình: {sum(final_stats[s]['bitrate'] if s in final_stats else 0 for s in streams):.1f} Mbps\n"
        f"Tổng số gói truyền lại: {final_stats['total_retr']}\n"
        f"Tỷ lệ công bằng: {fairness}"
    )

    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_img, dpi=300)
    plt.close()

# Phân tích và xuất báo cáo cho nhiều file
algos = [
    ('pfifo', 'ping_pfifo.txt'),
    ('red', 'ping_red.txt'),
    ('fq_codel', 'ping_fq_codel.txt'),
    ('codel', 'ping_codel.txt'),
    ('cake', 'ping_cake.txt')
]

for algo_name, fname in algos:
    print(f"\nPhân tích file: {fname}")
    data, final_stats = parse_pfifo_log(fname)
    plot_pfifo_analysis(data, final_stats, algo_name.upper(), f'{algo_name}_performance_analysis.png')
    report = {
        'Stream': ['5', '7', '9', '11', 'Tổng'],
        'Dữ liệu (MB)': [final_stats['stream5']['transfer'] if 'stream5' in final_stats else 0, 
                        final_stats['stream7']['transfer'] if 'stream7' in final_stats else 0,
                        final_stats['stream9']['transfer'] if 'stream9' in final_stats else 0,
                        final_stats['stream11']['transfer'] if 'stream11' in final_stats else 0,
                        sum(final_stats[s]['transfer'] if s in final_stats else 0 for s in ['stream5','stream7','stream9','stream11'])],
        'Bitrate (Mbps)': [final_stats['stream5']['bitrate'] if 'stream5' in final_stats else 0,
                          final_stats['stream7']['bitrate'] if 'stream7' in final_stats else 0,
                          final_stats['stream9']['bitrate'] if 'stream9' in final_stats else 0,
                          final_stats['stream11']['bitrate'] if 'stream11' in final_stats else 0,
                          sum(final_stats[s]['bitrate'] if s in final_stats else 0 for s in ['stream5','stream7','stream9','stream11'])],
        'Truyền lại': [final_stats['stream5']['total_retr'] if 'stream5' in final_stats else 0,
                      final_stats['stream7']['total_retr'] if 'stream7' in final_stats else 0,
                      final_stats['stream9']['total_retr'] if 'stream9' in final_stats else 0,
                      final_stats['stream11']['total_retr'] if 'stream11' in final_stats else 0,
                      final_stats['total_retr']]
    }
    df = pd.DataFrame(report)
    print(f"\nBáo Cáo Thống Kê Hiệu Năng {algo_name.upper()}:")
    print(df.to_string(index=False))
    out_csv = f'{algo_name}_performance_report.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nĐã lưu báo cáo ra file: {out_csv}")