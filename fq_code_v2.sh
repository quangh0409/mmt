# Thiết lập hàng đợi fq_codel
sudo tc qdisc replace dev enp0s3 root fq_codel \
    limit 2000 flows 2048 target 4ms interval 30ms quantum 1514 ecn

# Tạo tải hỗn hợp TCP/UDP
iperf3 -c 192.168.100.1 -t 60 -P 4 -u -b 10M -l 1000 --dscp 34 --logfile ./fq_codel/iperf_fq_codel.log &
iperf3 -c 192.168.100.1 -t 60 -P 4 -w 1m --logfile ./fq_codel/iperf_fq_codel_tcp.log

# Giám sát độ trễ
ping 192.168.100.1 -i 0.1 -w 60 -D > ./fq_codel/ping_fq_codel.log &

# Thu thập thống kê
for i in {1..12}; do
    date +"%T" >> ./fq_codel/fq_codel_stats.txt
    tc -s qdisc show dev enp0s3 >> ./fq_codel/fq_codel_stats.txt
    sleep 10
done

