# Thiết lập hàng đợi
sudo tc qdisc replace dev enp0s3 root pfifo limit 500 

# Chạy iperf3 với cấu hình nâng cao
iperf3 -c 192.168.100.1 -t 60 -P 4 -u -b 200M -l 1400 \
     --dscp 34 -A 0 --logfile ./pfifo/iperf_pfifo.log

# Giám sát độ trễ với kích thước gói lớn
ping 192.168.100.1 -i 0.1 -s 1000 -w 120 -Q 0x10 > ./pfifo/ping_pfifo.log &

# Thu thập thống kê định kỳ
for i in {1..12}; do
    date +"%T" >> ./pfifo/pfifo_stats_interval.txt
    tc -s qdisc show dev enp0s3 >> ./pfifo/pfifo_stats_interval.txt
    sleep 10
done
