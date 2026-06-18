#include <iostream>
#include <string>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <linux/rtnetlink.h>
#include <net/if.h>
#include <unistd.h>
#include <sys/select.h>
#include <nlohmann/json.hpp>

#include "../queue.h"

using json = nlohmann::json;

void start_network_collector(BoundedEventQueue& queue, const std::atomic<bool>& keep_running) {
    int nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (nl_fd < 0) {
        std::cerr << "[Sentinel] Network collector failed to create Netlink Route socket: " << std::strerror(errno) << std::endl;
        return;
    }

    struct sockaddr_nl sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.nl_family = AF_NETLINK;
    sa.nl_groups = RTMGRP_LINK | RTMGRP_IPV4_IFADDR;

    if (bind(nl_fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        std::cerr << "[Sentinel] Netlink Route bind failed: " << std::strerror(errno) << std::endl;
        close(nl_fd);
        return;
    }

    std::cout << "[Sentinel] Netlink Network Monitor active." << std::endl;

    char buf[4096];
    struct timeval tv;

    while (keep_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(nl_fd, &fds);

        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms timeout

        int ret = select(nl_fd + 1, &fds, nullptr, nullptr, &tv);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue; // Timeout, check keep_running

        ssize_t len = recv(nl_fd, buf, sizeof(buf), 0);
        if (len < 0) {
            if (errno == EAGAIN || errno == EINTR) continue;
            break;
        }

        struct nlmsghdr* nlh = (struct nlmsghdr*)buf;
        while (NLMSG_OK(nlh, len)) {
            if (nlh->nlmsg_type == NLMSG_DONE) break;
            
            if (nlh->nlmsg_type == RTM_NEWLINK || nlh->nlmsg_type == RTM_DELLINK) {
                struct ifinfomsg* ifi = (struct ifinfomsg*)NLMSG_DATA(nlh);
                char ifname[IF_NAMESIZE];
                if (if_indextoname(ifi->ifi_index, ifname) != nullptr) {
                    bool is_up = (ifi->ifi_flags & IFF_UP) && (ifi->ifi_flags & IFF_RUNNING);
                    
                    json j;
                    j["v"] = 1;
                    j["type"] = "NETWORK_CHANGE";
                    j["interface"] = ifname;
                    j["state"] = is_up ? "UP" : "DOWN";
                    queue.push(j.dump());
                }
            }
            else if (nlh->nlmsg_type == RTM_NEWADDR || nlh->nlmsg_type == RTM_DELADDR) {
                struct ifaddrmsg* ifa = (struct ifaddrmsg*)NLMSG_DATA(nlh);
                char ifname[IF_NAMESIZE];
                if (if_indextoname(ifa->ifa_index, ifname) != nullptr) {
                    json j;
                    j["v"] = 1;
                    j["type"] = "IP_CHANGE";
                    j["interface"] = ifname;
                    j["action"] = (nlh->nlmsg_type == RTM_NEWADDR) ? "ADDED" : "REMOVED";
                    queue.push(j.dump());
                }
            }
            nlh = NLMSG_NEXT(nlh, len);
        }
    }

    close(nl_fd);
}
