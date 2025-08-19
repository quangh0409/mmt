# Thiết lập hàng đợi RED
sudo tc qdisc replace dev enp0s3 root red \
    limit 600000 min 150000 max 300000 avpkt 1000 \
    burst 151 probability 0.02 ecn

# Tạo tải mạng
iperf3 -c 192.168.100.1 -t 60 -P 4 -u -b 100M -l 1200 \
    -O 5 --dscp 46 --logfile ./red/iperf_red.log

# Giám sát độ trễ
ping 192.168.100.1 -i 0.1 -s 1000 -w 60 -Q 0x10 > ./red/ping_red.log &

# Thu thập thống kê định kỳ
for i in {1..12}; do
    date +"%T" >> ./red/red_stats.txt
    tc -s qdisc show dev enp0s3 >> ./red/red_stats.txt
    sleep 10
done
