# Thiết lập hàng đợi CoDel
sudo tc qdisc replace dev enp0s3 root codel \
    limit 500 target 10ms interval 80ms ecn

# Tạo tải UDP nhạy cảm độ trễ
iperf3 -c 192.168.100.1 -t 60 -P 4 -u -b 20M -l 200 -w 200k \
    -O 5 --fq-rate 15M --logfile ./codel/iperf_codel.log

# Giám sát độ trễ VoIP
ping 192.168.100.1 -i 0.1 -w 60 -s 120 > ./codel/ping_codel.log &

# Thu thập thống kê chi tiết
for i in {1..12}; do
    date +"%T" >> ./codel/codel_stats.txt
    tc -s qdisc show dev enp0s3 >> ./codel/codel_stats.txt
    sleep 10
done
