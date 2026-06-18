#include <iostream>
#include <string>
#include <atomic>
#include <cstring>
#include <libudev.h>
#include <unistd.h>
#include <sys/select.h>
#include <nlohmann/json.hpp>

#include "../queue.h"

using json = nlohmann::json;

void start_device_collector(BoundedEventQueue& queue, const std::atomic<bool>& keep_running) {
    struct udev* udev = udev_new();
    if (!udev) {
        std::cerr << "[Sentinel] udev_new() failed." << std::endl;
        return;
    }

    struct udev_monitor* mon = udev_monitor_new_from_netlink(udev, "udev");
    if (!mon) {
        std::cerr << "[Sentinel] udev_monitor_new_from_netlink() failed." << std::endl;
        udev_unref(udev);
        return;
    }

    // Monitor both USB interfaces and external storage block devices
    udev_monitor_filter_add_match_subsystem_devtype(mon, "usb", nullptr);
    udev_monitor_filter_add_match_subsystem_devtype(mon, "block", nullptr);

    if (udev_monitor_enable_receiving(mon) < 0) {
        std::cerr << "[Sentinel] Failed to enable udev receiving." << std::endl;
        udev_monitor_unref(mon);
        udev_unref(udev);
        return;
    }

    int fd = udev_monitor_get_fd(mon);
    std::cout << "[Sentinel] udev Device Monitor active." << std::endl;

    struct timeval tv;

    while (keep_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);

        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms timeout

        int ret = select(fd + 1, &fds, nullptr, nullptr, &tv);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue; // Timeout, check keep_running

        struct udev_device* dev = udev_monitor_receive_device(mon);
        if (dev) {
            const char* action = udev_device_get_action(dev);
            const char* devnode = udev_device_get_devnode(dev);
            const char* subsystem = udev_device_get_subsystem(dev);
            const char* devtype = udev_device_get_devtype(dev);

            json j;
            j["v"] = 1;
            j["type"] = "DEVICE_CHANGE";
            j["action"] = action ? action : "";
            j["devnode"] = devnode ? devnode : "";
            j["subsystem"] = subsystem ? subsystem : "";
            j["devtype"] = devtype ? devtype : "";
            queue.push(j.dump());

            udev_device_unref(dev);
        }
    }

    udev_monitor_unref(mon);
    udev_unref(udev);
}
