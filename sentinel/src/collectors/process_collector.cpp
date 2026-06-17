#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <linux/netlink.h>
#include <linux/connector.h>
#include <linux/cn_proc.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include "../queue.h"

using json = nlohmann::json;

static std::string get_process_name(pid_t pid) {
    std::ifstream comm_file("/proc/" + std::to_string(pid) + "/comm");
    std::string name;
    if (comm_file >> name) {
        return name;
    }
    return "unknown";
}

void start_process_collector(BoundedEventQueue& queue, const std::atomic<bool>& keep_running) {
    int nl_fd = socket(PF_NETLINK, SOCK_DGRAM, NETLINK_CONNECTOR);
    if (nl_fd < 0) {
        std::cerr << "[Sentinel] Process collector failed to create Netlink socket. Ensure CAP_NET_ADMIN is set." << std::endl;
        return;
    }

    struct sockaddr_nl sa_nl;
    std::memset(&sa_nl, 0, sizeof(sa_nl));
    sa_nl.nl_family = AF_NETLINK;
    sa_nl.nl_groups = CN_IDX_PROC;
    sa_nl.nl_pid = getpid();

    if (bind(nl_fd, (struct sockaddr*)&sa_nl, sizeof(sa_nl)) < 0) {
        std::cerr << "[Sentinel] Netlink bind failed: " << std::strerror(errno) << std::endl;
        close(nl_fd);
        return;
    }

    // Subscribe to process events
    char send_buf[1024];
    std::memset(send_buf, 0, sizeof(send_buf));

    struct nlmsghdr* nlhdr = (struct nlmsghdr*)send_buf;
    struct cn_msg* cnmsg = (struct cn_msg*)NLMSG_DATA(nlhdr);
    enum proc_cn_mcast_op* op = (enum proc_cn_mcast_op*)cnmsg->data;

    nlhdr->nlmsg_len = NLMSG_LENGTH(sizeof(struct cn_msg) + sizeof(enum proc_cn_mcast_op));
    nlhdr->nlmsg_pid = getpid();
    nlhdr->nlmsg_type = NLMSG_DONE;

    cnmsg->id.idx = CN_IDX_PROC;
    cnmsg->id.val = CN_VAL_PROC;
    cnmsg->len = sizeof(enum proc_cn_mcast_op);

    *op = PROC_CN_MCAST_LISTEN;

    if (send(nl_fd, nlhdr, nlhdr->nlmsg_len, 0) < 0) {
        std::cerr << "[Sentinel] Netlink send failed: " << std::strerror(errno) << std::endl;
        close(nl_fd);
        return;
    }

    std::cout << "[Sentinel] Netlink Process Connector active." << std::endl;

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

        struct nlmsghdr* nlhdr = (struct nlmsghdr*)buf;
        while (NLMSG_OK(nlhdr, len)) {
            if (nlhdr->nlmsg_type == NLMSG_ERROR || nlhdr->nlmsg_type == NLMSG_NOOP) {
                nlhdr = NLMSG_NEXT(nlhdr, len);
                continue;
            }

            struct cn_msg* cnmsg = (struct cn_msg*)NLMSG_DATA(nlhdr);
            if (cnmsg->id.idx == CN_IDX_PROC && cnmsg->id.val == CN_VAL_PROC) {
                struct proc_event* ev = (struct proc_event*)cnmsg->data;
                
                if (ev->what == PROC_EVENT_EXEC) {
                    pid_t pid = ev->event_data.exec.process_pid;
                    // Filter out kernel threads (PIDs < 100)
                    if (pid >= 100) {
                        std::string name = get_process_name(pid);
                        if (name != "unknown" && name != "sentinel-daemon") {
                            json j;
                            j["v"] = 1;
                            j["type"] = "PROCESS_START";
                            j["pid"] = pid;
                            j["name"] = name;
                            queue.push(j.dump());
                        }
                    }
                }
                else if (ev->what == PROC_EVENT_EXIT) {
                    pid_t pid = ev->event_data.exit.process_pid;
                    if (pid >= 100) {
                        json j;
                        j["v"] = 1;
                        j["type"] = "PROCESS_EXIT";
                        j["pid"] = pid;
                        queue.push(j.dump());
                    }
                }
            }
            nlhdr = NLMSG_NEXT(nlhdr, len);
        }
    }

    close(nl_fd);
}
