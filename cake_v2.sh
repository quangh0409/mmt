# Thiết lập hàng đợi CAKE
sudo tc qdisc replace dev enp0s3 root cake \
    bandwidth 500Mbit diffserv4 dual-dsthost nat nowash \
    overhead 38 rtt 50ms memlimit 100Mb ack-filter

# Tạo tải hỗn hợp phức tạp
iperf3 -c 192.168.100.1 -t 60 -P 4 -u -b 30M -l 800 --dscp 46 --logfile ./cake/iperf_cake_voip.log &
iperf3 -c 192.168.100.1 -t 60 -P 4 -w 2m --logfile ./cake/iperf_cake_tcp.log &
iperf3 -c 192.168.100.1 -t 60 -P 4 -u -b 200M -l 1400 --dscp 0 --logfile ./cake/iperf_cake_bulk.log

# Giám sát toàn diện
ping 192.168.100.1 -i 0.1 -w 60 > ./cake/ping_cake.log &
for i in {1..12}; do
    date +"%T" >> ./cake/cake_stats.txt
    tc -s qdisc show dev enp0s3 >> ./cake/cake_stats.txt
    sleep 10
done
